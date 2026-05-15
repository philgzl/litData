import hashlib
import os
import random
from abc import abstractmethod
from collections.abc import Iterator, Sequence
from copy import deepcopy
from fnmatch import fnmatch
from typing import Any

import torch
from torch.utils.data import IterableDataset

from litdata.streaming import Cache
from litdata.streaming.item_loader import ParquetLoader
from litdata.streaming.resolver import _resolve_dir
from litdata.streaming.sampler import ChunkedIndex
from litdata.utilities.hf_dataset import index_hf_dataset

__SAMPLE_KEY__ = "__SAMPLE__"
__WORKER_ID_KEY__ = "__WORKER_ID__"
__WORKER_STATE_KEY__ = "__WORKER_STATE__"


class BaseStreamingDataset(IterableDataset):
    """Base class for streaming datasets."""

    length: int
    resume: bool
    use_dataloader: bool
    _worker_states: dict[int, dict[str, Any]]
    _is_last: bool | None

    @abstractmethod
    def _initialize_worker_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def _iterate(
        self,
        worker_id: int,
        num_workers: int,
        worker_length: int,
        worker_state: dict[str, Any],
    ) -> Iterator[Any]: ...

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the dataset."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if worker_id not in self._worker_states or not self.resume:
            self._worker_states[worker_id] = self._initialize_worker_state()

        worker_length = -1 if self.length < 0 else self.length // num_workers + (worker_id < self.length % num_workers)
        worker_state = self._worker_states[worker_id]

        for item in self._iterate(worker_id, num_workers, worker_length, worker_state):
            if self.use_dataloader:
                yield {
                    __SAMPLE_KEY__: item,
                    __WORKER_ID_KEY__: worker_id,
                    __WORKER_STATE_KEY__: deepcopy(worker_state) if worker_info is None else worker_state,
                }
            else:
                yield item

    def _get_is_last(self, worker_length: int, items_yielded: int) -> bool:
        if self._is_last is not None:
            return self._is_last
        return worker_length >= 0 and (items_yielded + 1) >= worker_length

    def __len__(self) -> int:
        """Return the number of items to yield per epoch."""
        if self.length < 0:
            raise TypeError("Length is negative. Dataset is infinite.")
        return self.length

    def state_dict(self) -> dict[str, Any]:
        """Return the dataset state."""
        return deepcopy(self._worker_states)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the dataset state."""
        self._worker_states = deepcopy(state_dict)


class HFStreamingDataset(BaseStreamingDataset):
    """Hugging Face streaming dataset."""

    def __init__(
        self,
        hf_dataset_url: str,
        length: int | None = None,
        resume: bool = True,
        cache_dir: str | None = None,
        max_cache_size: int | str = "100GB",
        max_pre_download: int = 2,
        seed: int = 0,
    ) -> None:
        """Initialize the Hugging Face streaming dataset.

        Args:
            hf_dataset_url: HuggingFace dataset URL. Must end with ".parquet" and can include glob patterns.
            length: Number of items to yield per epoch. If greater than the number of rows, the dataset is cycled.
            resume: Whether to resume from where we left off when calling `__iter__` again.
            cache_dir: Local directory for caching downloaded files.
            max_cache_size: Maximum cache size.
            max_pre_download: Number of chunks to pre-download.
            seed: Random seed for shuffling chunks.
        """
        super().__init__()

        if not hf_dataset_url.startswith("hf://"):
            raise ValueError(f"Expected hf:// URL, got: {hf_dataset_url}")
        if not hf_dataset_url.endswith(".parquet"):
            raise ValueError(f"Expected URL to end with .parquet, got: {hf_dataset_url}")

        hf_dataset_url, self.fnmatch_pattern = os.path.split(hf_dataset_url)
        self.input_dir = _resolve_dir(hf_dataset_url)
        self.cache_dir = _resolve_dir(cache_dir)
        self.input_dir.path = index_hf_dataset(dataset_url=hf_dataset_url, cache_dir=self.cache_dir.path)
        self.item_loader = ParquetLoader()

        self.max_cache_size = max_cache_size
        self.max_pre_download = max_pre_download
        self.seed = seed

        self._cache: Cache | None = None
        self._filtered_chunks: list[int] | None = None

        self.length = self.num_rows() if length is None else length
        self.resume = resume
        self.use_dataloader = False
        self._worker_states: dict[int, dict[str, Any]] = {}
        self._is_last = None

    def num_rows(self) -> int:
        """Return the number of rows in the dataset. Note this is not the same as `__len__`."""
        all_chunks = self._get_cache()._reader.config._chunks
        filtered_indices = self._get_filtered_chunk_indices()
        return sum(all_chunks[idx]["chunk_size"] for idx in filtered_indices)

    def _initialize_worker_state(self) -> dict[str, Any]:
        return {"chunk_idx": 0, "item_idx": 0, "cycle_idx": 0}

    def _iterate(
        self,
        worker_id: int,
        num_workers: int,
        worker_length: int,
        worker_state: dict[str, Any],
    ) -> Iterator[Any]:
        cache = self._get_cache()

        # Get filtered chunks based on fnmatch pattern
        filtered_chunk_indices = self._get_filtered_chunk_indices()

        # Calculate which chunks belong to this worker
        worker_chunks = [idx for i, idx in enumerate(filtered_chunk_indices) if i % num_workers == worker_id]

        # Iterate until we've yielded enough items
        items_yielded = 0
        while worker_length < 0 or items_yielded < worker_length:

            # Shuffle chunks at the start of each cycle
            shuffled_chunks = self._shuffle_chunks(worker_chunks, worker_id, worker_state["cycle_idx"])

            # Iterate through chunks starting from where we left off
            for shuffled_chunk_idx in range(worker_state["chunk_idx"], len(shuffled_chunks)):
                chunk_idx = shuffled_chunks[shuffled_chunk_idx]
                chunk_info = cache._reader.config._chunks[chunk_idx]
                chunk_size = chunk_info["chunk_size"]

                # Get the global index offset for this chunk
                chunk_begin = cache._reader.config._intervals[chunk_idx][0]

                # Iterate through items in this chunk
                for item_idx in range(worker_state["item_idx"], chunk_size):

                    # Check if this is the last item this worker will yield
                    is_last = self._get_is_last(worker_length, items_yielded)

                    # Calculate global index for this item
                    global_index = chunk_begin + item_idx

                    # Load item using ChunkedIndex
                    chunked_index = ChunkedIndex(
                        index=global_index,
                        chunk_index=chunk_idx,
                        chunk_indexes=shuffled_chunks,
                        is_last_index=is_last,
                        chunk_size=chunk_size,
                    )
                    item = cache[chunked_index]

                    # Update state and counter before yielding
                    if item_idx + 1 >= chunk_size:
                        worker_state["item_idx"] = 0
                        if shuffled_chunk_idx + 1 >= len(shuffled_chunks):
                            worker_state["chunk_idx"] = 0
                            worker_state["cycle_idx"] += 1
                        else:
                            worker_state["chunk_idx"] += 1
                    else:
                        worker_state["item_idx"] = item_idx + 1
                    items_yielded += 1
                    yield item

                    if is_last:
                        return

    def _create_cache(self) -> Cache:
        cache = Cache(
            input_dir=self.input_dir,
            subsampled_files=None,
            region_of_interest=None,
            item_loader=self.item_loader,
            chunk_bytes=1,
            serializers=None,
            max_cache_size=self.max_cache_size,
            encryption=None,
            max_pre_download=self.max_pre_download,
            on_demand_bytes=False,
        )
        cache._reader._try_load_config()
        if not cache.filled:
            raise ValueError(f"Dataset at {self.input_dir} doesn't contain valid index.")
        return cache

    def _get_cache(self) -> Cache:
        if self._cache is None:
            self._cache = self._create_cache()
        return self._cache

    def _get_filtered_chunk_indices(self) -> list[int]:
        if self._filtered_chunks is None:
            cache = self._get_cache()
            all_chunks = cache._reader.config._chunks
            self._filtered_chunks = [
                idx for idx, chunk in enumerate(all_chunks) if fnmatch(chunk["filename"], self.fnmatch_pattern)
            ]
            assert self._filtered_chunks, f"No chunks matched the pattern '{self.fnmatch_pattern}'."
        return self._filtered_chunks

    def _shuffle_chunks(self, chunks: list[int], worker_id: int, cycle_idx: int) -> list[int]:
        chunks = chunks.copy()
        # Deterministic hash for stable and reproducible shuffling across runs
        seed_bytes = f"{self.seed}_{worker_id}_{cycle_idx}".encode()
        seed_int = int(hashlib.md5(seed_bytes).hexdigest(), 16) % (2**32)
        random.Random(seed_int).shuffle(chunks)
        return chunks


class ParallelStreamingDataset(BaseStreamingDataset):
    def __init__(self, datasets: Sequence[BaseStreamingDataset], length: int, resume: bool = False) -> None:
        super().__init__()
        self.datasets = datasets
        self.length = length
        self.resume = resume
        self.use_dataloader = False
        self._worker_states: dict[int, dict[str, Any]] = {}
        self._is_last = None

    def _initialize_worker_state(self) -> dict[str, Any]:
        return {idx: dataset._initialize_worker_state() for idx, dataset in enumerate(self.datasets)}

    def _iterate(
        self,
        worker_id: int,
        num_workers: int,
        worker_length: int,
        worker_state: dict[str, Any],
    ) -> Iterator[Any]:
        iterators = [
            dataset._iterate(worker_id, num_workers, worker_length, worker_state[idx])
            for idx, dataset in enumerate(self.datasets)
        ]
        items_yielded = 0
        while worker_length < 0 or items_yielded < worker_length:
            for dataset in self.datasets:
                dataset._is_last = self._get_is_last(worker_length, items_yielded)
            items = [next(it) for it in iterators]
            items_yielded += 1
            yield items


class CombinedStreamingDataset(BaseStreamingDataset):
    def __init__(self, datasets: Sequence[BaseStreamingDataset], length: int, resume: bool = False) -> None:
        super().__init__()
        self.datasets = datasets
        self.length = length
        self.resume = resume
        self.use_dataloader = False
        self._worker_states: dict[int, dict[str, Any]] = {}
        self._is_last = None

    def _initialize_worker_state(self) -> dict[str, Any]:
        return {
            **{idx: dataset._initialize_worker_state() for idx, dataset in enumerate(self.datasets)},
            "dataset_idx": 0,
        }

    def _iterate(
        self,
        worker_id: int,
        num_workers: int,
        worker_length: int,
        worker_state: dict[str, Any],
    ) -> Iterator[Any]:
        iterators = [
            dataset._iterate(worker_id, num_workers, worker_length, worker_state[idx])
            for idx, dataset in enumerate(self.datasets)
        ]
        items_yielded = 0
        while worker_length < 0 or items_yielded < worker_length:
            for dataset, iterator in zip(self.datasets, iterators):
                dataset._is_last = self._get_is_last(worker_length, items_yielded)
                item = next(iterator)
                worker_state["dataset_idx"] = (worker_state["dataset_idx"] + 1) % len(self.datasets)
                items_yielded += 1
                yield item


class StreamingDataloader(torch.utils.data.DataLoader):
    """DataLoader wrapper for BaseStreamingDataset instances."""

    def __init__(self, dataset: BaseStreamingDataset, *args, **kwargs):
        """Initialize the streaming DataLoader."""
        if not isinstance(dataset, BaseStreamingDataset):
            raise ValueError(f"Dataset must be an instance of BaseStreamingDataset. Got {type(dataset).__name__}")
        super().__init__(dataset, *args, collate_fn=self.collate_fn, **kwargs)
        self._worker_states: dict[int, dict[str, Any]] = {}

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the DataLoader."""
        self.dataset.use_dataloader = True
        self.dataset.load_state_dict(self._worker_states)
        for batch in super().__iter__():
            worker_ids = batch[__WORKER_ID_KEY__]
            worker_states = batch[__WORKER_STATE_KEY__]
            assert all(worker_id == worker_ids[0] for worker_id in worker_ids)
            self._worker_states[worker_ids[0]] = worker_states[-1]
            yield batch[__SAMPLE_KEY__]
        self.dataset.use_dataloader = False

    def state_dict(self) -> dict[str, Any]:
        """Return the DataLoader state."""
        return deepcopy(self._worker_states)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the DataLoader state."""
        self.dataset.load_state_dict(state_dict)
        self._worker_states = state_dict

    @staticmethod
    def collate_fn(batch: list[Any]) -> Any:
        """Collate function to combine samples into a batch."""
        samples = [item[__SAMPLE_KEY__] for item in batch]
        worker_ids = [item[__WORKER_ID_KEY__] for item in batch]
        worker_states = [item[__WORKER_STATE_KEY__] for item in batch]
        return {
            __SAMPLE_KEY__: torch.utils.data.default_collate(samples),
            __WORKER_ID_KEY__: worker_ids,
            __WORKER_STATE_KEY__: worker_states,
        }
