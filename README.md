The code will be available soon...

<p align="center">
<img src="https://github.com/ci-ber/PHANES/assets/106509806/fc0ac560-2668-444a-bab9-a5e90eddf812" width="200" class="center">
</p>
<h1 align="center">
  <br>
Reversing the Abnormal: Pseudo-Healthy Generative Networks for Anomaly Detection
  <br>
</h1>
</h1>
  <p align="center">
    <a href="https://ci.bercea.net">Cosmin Bercea</a> •
    <a href="https://www.neurokopfzentrum.med.tum.de/neuroradiologie/mitarbeiter-profil-wiestler.html">Benedikt Wiestler</a> •
    <a href="https://aim-lab.io/author/daniel-ruckert/">Daniel Rueckert </a> •
    <a href="https://compai-lab.github.io/author/julia-a.-schnabel/">Julia A. Schnabel </a>
  </p>
<h4 align="center">Official repository of the paper</h4>
<h4 align="center">MICCAI 2023</h4>
<h4 align="center"><a href="https://ci.bercea.net/project/phanes/">Project Website</a> • <a href="https://arxiv.org/pdf/2303.08452.pdf">Preprint</a> </h4>

<p align="center">
<img src="https://github.com/ci-ber/PHANES/assets/106509806/14959ea5-b111-4967-8741-d24b009d3c32">
</p>

## Citation

If you find our work helpful, please cite our paper:
```
@article{bercea2023reversing,
  title={Reversing the abnormal: Pseudo-healthy generative networks for anomaly detection},
  author={Bercea, Cosmin I and Wiestler, Benedikt and Rueckert, Daniel and Schnabel, Julia A},
  journal={arXiv preprint arXiv:2303.08452},
  year={2023}
}
```

> **Abstract:** *Early and accurate disease detection is crucial for patient management and successful treatment outcomes. However, the automatic identification of anomalies in medical images can be challenging. Conventional methods rely on large labeled datasets which are difficult to obtain. To overcome these limitations, we introduce a novel unsupervised approach, called PHANES (Pseudo Healthy generative networks for ANomaly Segmentation). Our method has the capability of reversing anomalies, i.e., preserving healthy tissue and replacing anomalous regions with pseudo-healthy (PH) reconstructions. Unlike recent diffusion models, our method does not rely on a learned noise distribution nor does it introduce random alterations to the entire image. Instead, we use latent generative networks to create masks around possible anomalies, which are refined using inpainting generative networks. We demonstrate the effectiveness of PHANES in detecting stroke lesions in T1w brain MRI datasets and show significant improvements over state-of-the-art (SOTA) methods. We believe that our proposed framework will open new avenues for interpretable, fast, and accurate anomaly segmentation with the potential to support various clinical-oriented downstream tasks.*


## Setup and Run

The code is based on the deep learning framework from the Institute of Machine Learning in Biomedical Imaging: https://github.com/compai-lab/iml-dl

### Framework Overview: 

<p align="center">
<img src="https://github.com/ci-ber/PHANES/assets/106509806/a298aa9d-9163-4df4-a47d-1124b6d82724">
</p>

#### 1). Set up wandb (https://docs.wandb.ai/quickstart)

Sign up for a free account and login to your wandb account.
```bash
wandb login
```
Paste the API key from https://wandb.ai/authorize when prompted.

#### 2). Clone repository

```bash
git clone https://github.com/ci-ber/PHANES.git
cd PHANES
```

#### 3). Install requirements

```bash
pip install -r requirements.txt
```

#### 4). Install PyTorch 

> Example installation:

* *with cuda*: 
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
* *w/o cuda*:
```
pip3 install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 5). Download datasets 

<h4 align="center"><a href="https://brain-development.org/ixi-dataset/">IXI</a> • <a href="https://fastmri.org">FastMRI</a> • <a href="https://github.com/microsoft/fastmri-plus"> Labels for FastMRI</a> • <a href="https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html">Atlas (Stroke) </a> </h4>

> *Alternatively you can use your own mid-axial slices of T1w brain scans with our <a href=""> pre-trained weights</a> or train from scratch on other anatomies and modalities.*


#### 6). Run the pipeline

Run the main script with the corresponding config like this:

```bash
python core/Main.py --config_path ./projects/phanes/phanes.yaml
```

Refer to phanes.yaml for the default configuration.

# That's it, enjoy! :rocket:






