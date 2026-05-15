from dotenv import load_dotenv
from tqdm import tqdm

from custom_streaming_dataset import CombinedStreamingDataset, HFStreamingDataset, ParallelStreamingDataset, StreamingDataloader

load_dotenv()

dset_1 = HFStreamingDataset("hf://datasets/philgzl/clarity/data/*.parquet", length=-1)
dset_2 = HFStreamingDataset("hf://datasets/philgzl/vctk/data/*.parquet", length=-1)

dset_par = ParallelStreamingDataset([dset_1, dset_2], length=64)

dloader = StreamingDataloader(dset_par, batch_size=4, num_workers=4)

# for x in tqdm(dloader):
#     print(x[1]["name"])

dset_com = CombinedStreamingDataset([dset_1, dset_2], length=64)

dloader = StreamingDataloader(dset_com, batch_size=4, num_workers=4)

for x in tqdm(dloader):
    print(x["name"])
