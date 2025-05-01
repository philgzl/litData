# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from litdata.streaming.dataset import StreamingDataset
from litdata.utilities.base import (
    __NUM_CYCLES_KEY__,
    __NUM_SAMPLES_YIELDED_KEY__,
    __SAMPLES_KEY__,
    _BaseDatasetWrapperIterator,
    _BaseStreamingDatasetWrapper,
)
from litdata.utilities.env import _WorkerEnv

logger = logging.getLogger("litdata.streaming.parallel")


class ParallelStreamingDataset(_BaseStreamingDatasetWrapper):
    """Enables to stream data from multiple StreamingDataset in parallel.

    The yielded samples are tuples where the `n`-th element is a sample from the `n`-th dataset.

    Additionally, the parallel dataset keeps track of the number of samples fetched to enable reusability of the
    datasets.

    The parallel dataset can be configured to raise a ``StopIteration`` as soon as any of the datasets is exhausted, or
    to cycle through the datasets until a given number of samples are yielded.
    """

    def __init__(
        self,
        datasets: List[StreamingDataset],
        length: Optional[int | float] = None,
        force_override_state_dict: bool = False,
    ) -> None:
        """Enable to stream data from multiple StreamingDataset in parallel.

        Args:
            datasets: The list of the StreamingDataset to use.
            length: The number of samples to yield. If ``None``, the datasets are iterated over until one of them is
                exhausted. If an integer, the datasets are cycled until ``length`` samples are yielded. Can be a
                ``float("inf")`` for an infinite dataset.
            force_override_state_dict: Boolean flag for allowing local arguments to override a loaded state dict.

        """
        self._check_datasets(datasets)

        if length is not None and not isinstance(length, int) and length != float("inf"):
            raise ValueError(f"`length` must be `None`, an integer, or `float('inf')`, got {length}.")

        self._datasets = datasets
        self._length = length
        self._force_override_state_dict = force_override_state_dict
        self._iterator: Optional[_ParallelDatasetIterator] = None
        self._use_streaming_dataloader = False
        self._num_samples_yielded: Optional[List[int]] = None
        self._num_cycles: Optional[List[int]] = None
        self._current_epoch = 0
        self.num_workers = 1
        self.batch_size = 1

        if length is not None:
            for dataset in self._datasets:
                if isinstance(dataset, StreamingDataset):
                    dataset.set_epoch(1)

    def set_epoch(self, current_epoch: int) -> None:
        self._current_epoch = current_epoch
        if self._length is not None:
            # do not set the epoch as cycling datasets have their own epoch counter
            return
        for dataset in self._datasets:
            dataset.set_epoch(current_epoch)

    def get_len(self, num_workers: int, batch_size: int) -> Optional[int]:
        self.num_workers = num_workers
        self.batch_size = batch_size
        lengths = [self._get_len(d) for d in self._datasets]
        if self._length is None:
            return min(lengths)
        if self._length == float("inf"):
            return None
        if isinstance(self._length, int):
            return self._length
        raise ValueError(f"Invalid ParallelStreamingDataset _length attribute: {self._length}.")

    def __iter__(self) -> Iterator[Any]:
        worker_env = _WorkerEnv.detect()

        num_samples_yielded = None
        if self._num_samples_yielded is not None and worker_env.rank in self._num_samples_yielded:
            num_samples_yielded = self._num_samples_yielded.get(worker_env.rank, 0)

        num_cycles = None
        if self._num_cycles is not None and worker_env.rank in self._num_cycles:
            num_cycles = self._num_cycles.get(worker_env.rank, 0)

        length = self._length
        if length not in [None, float("inf")]:
            length = self._length // worker_env.world_size + (worker_env.rank < self._length % worker_env.world_size)

        dset_lengths = [
            self._get_len(d) // worker_env.world_size + (worker_env.rank < self._get_len(d) % worker_env.world_size)
            for d in self._datasets
        ]

        self._iterator = _ParallelDatasetIterator(
            self._datasets, self._use_streaming_dataloader, num_samples_yielded, num_cycles, length, dset_lengths
        )
        return self._iterator

    def __len__(self) -> int:
        return self.get_len(self.num_workers, self.batch_size if self.batch_size else 1)

    def _get_num_samples_yielded(
        self, num_samples_yielded: Dict[int, List[int]], num_cycles: Dict[int, List[int]]
    ) -> Tuple[List[int], List[int]]:
        assert num_samples_yielded.keys() == num_cycles.keys()
        assert all(len(s) == len(c) for s, c in zip(num_samples_yielded.values(), num_cycles.values()))
        output = [0 for _ in range(len(self._datasets))]
        cycles = [0 for _ in range(len(self._datasets))]
        for i, (num_samples_yielded, num_cycles) in enumerate(
            zip(zip(*num_samples_yielded.values()), zip(*num_cycles.values()))
        ):
            cycles[i] = max(num_cycles)
            output[i] = sum(s for (s, c) in zip(num_samples_yielded, num_cycles) if c == cycles[i])
        return output, cycles

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self._use_streaming_dataloader:
            self._num_cycles = state_dict["num_cycles"]


class _ParallelDatasetIterator(_BaseDatasetWrapperIterator):
    def __init__(
        self,
        datasets: List[StreamingDataset],
        use_streaming_dataloader: bool,
        num_samples_yielded: Any,
        num_cycles: Any,
        length: Optional[int | float],
        dset_lengths: List[int],
    ) -> None:
        self._datasets = datasets
        self._dataset_iters = [iter(dataset) for dataset in datasets]
        self._num_samples_yielded = num_samples_yielded or [0 for _ in range(len(datasets))]
        self._num_cycles = num_cycles or [0 for _ in range(len(datasets))]
        self._length = length
        self._use_streaming_dataloader = use_streaming_dataloader
        if self._length in [None, float("inf"), 0]:
            self._count = 0
        else:
            self._count = (dset_lengths[0] * self._num_cycles[0] + self._num_samples_yielded[0]) % self._length
            assert all(
                (dset_lengths[i] * self._num_cycles[i] + self._num_samples_yielded[i]) % self._length == self._count
                for i in range(1, len(dset_lengths))
            )

    def __next__(self) -> Tuple[Any]:
        if self._length is not None and self._count >= self._length:
            raise StopIteration
        samples, _resets = zip(*[self._get_sample(i) for i in range(len(self._datasets))])
        # update _num_samples_yielded and _num_cycles only if samples were successfully fetched from all datasets
        for i, _reset in enumerate(_resets):
            self._num_samples_yielded[i] = 1 if _reset else self._num_samples_yielded[i] + 1
            self._num_cycles[i] = self._num_cycles[i] + 1 if _reset else self._num_cycles[i]
        self._count += 1
        if self._use_streaming_dataloader:
            return {
                __SAMPLES_KEY__: samples,
                __NUM_SAMPLES_YIELDED_KEY__: self._num_samples_yielded,
                __NUM_CYCLES_KEY__: self._num_cycles,
            }
        return samples

    def _get_sample(self, dataset_index: int) -> Tuple[Any, bool]:
        _reset = False
        try:
            sample = next(self._dataset_iters[dataset_index])
        except StopIteration as e:
            if self._length is None:
                raise e
            self._dataset_iters[dataset_index] = iter(self._datasets[dataset_index])
            _reset = True
            try:
                sample = next(self._dataset_iters[dataset_index])
            except StopIteration:
                raise RuntimeError("Failed to get sample from dataset after cycling. Is the dataset empty?")
        return sample, _reset
