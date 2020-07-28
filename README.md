## CShaper

Implementation for in *Establishment of morphological atlas of Caenorhabditis elegans embryo using deep-learning-based 4D
 segmentation*, by Jianfeng Cao, Guoye Guan, Vincy Wing Sze Ho, Ming-Kin Wong, Lu-Yan Chan, Chao Tang, Zhongying Zhao, & Hong Yan.

### Usage
This implementation is based on Tensorflow and python3.6, trained with one GPU NVIDIA 2080Ti oon Linux. Steps for training
and testing are listed as below.
* **Intsall dependency library**:
```buildoutcfg
    pip install requirements.txt
```
* **Train**: Download the data from this link (TBD) and put it into `./Data` folder, Set parameters
in `./ConfigMemb/train_edt_discrete.txt`, then run
    ```buildoutcfg
    python train.py ./ConfigMemb/train_edt_discrete.txt
    ```
* **Test**: Put the raw data (membane and nucleus stack, and CD files from [AceTree](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1501046/))
into `./Data/MembValidation/`. Example data is also available through previous data link. Set parameters in 
`./ConfigMemb/test_edt_discrete.txt` and run
    ```buildoutcfg
    python test_edt.py ./ConfigMemb/test_edt_discrete.txt
    ```
    Then binary membrane and initial segmented cell are saved in `./ResultCell/BothWithRandomnet` and
    `BothWithRandomnetPostseg`, respectively. To unify the label of cell based on AceTree file,
    run 
    ```buildoutcfg
    python shape_analysis.py ./ConfigMemb/shape_config.txt
    ```
* **Structure of folders**: (Folders and files in `.gitignore` are not shown in this repository)
    ```buildoutcfg
    DMapNet is used to segmented membrane stack of C. elegans at cellular level
    DMapNet/
      |--configmemb/: parameters for training, testing and unifying label
      |--Data/: raw membrane, raw nucleus and AceTree file (CD**.csv)
          |--MembTraining/: image data with manual annotations
          |--MembValidation/: image data to be segmented
      |--ModelCell/: trained models 
      |--ResultCell/: Segmentation result
          |--BothWithRandomnet/: Binary membrane segmentation from DMapNet
          |--BothWithRandomnetPostseg/: segmented cell before and after label unifying
          |--NucleusLoc/: nucleus location information and annotation
          |--StatShape/: cell lineage tree (with time duration)
      |--ShapeUtil/: utils for unifying cells and calculating robustness
          |--AceForLabel/: multiple AceTree files for generating namedictionary
          |--RobustStat/: nucleus lost sration and cell surface...
          |--TemCellGraph/: temporary result for calculating surface, volume...
        
      |--Util/: utils for training and testing
    ```
### Related
* Project file for CellProfiler involved in evaluation ([link](https://portland-my.sharepoint.com/:u:/g/personal/jfcao3-c_ad_cityu_edu_hk/ETN3Z6j4TklAko6NvQDIujwBwzoixX26EajSOaoeeme2jg?download=1)).
* Parameter files for RACE ([link](https://portland-my.sharepoint.com/:u:/g/personal/jfcao3-c_ad_cityu_edu_hk/EX_iCNByGBtMlZI7G8bRgSMBqNfaCdAbq3cHDrGc-k6d5Q?download=1)). 


### Acknowledgement
We thank [brats17](https://github.com/taigw/brats17) which we referred to when implementing the code.
