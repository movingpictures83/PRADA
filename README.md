# PRADA

## Prioritization of Regulatory Pathways based on Analysis of RNA Dynamics Alterations

Dysregulation of RNA stability plays an important role in cancer progression. Key regulators of RNA turnover, such as miRNAs and RNA-binding proteins, have been implicated in a variety of cancers - however, the list of annotated regulatory programs that govern the RNA lifecycle remains incomplete. The development of analytical frameworks for systematic discovery of post-transcriptional regulators is critical for a better understanding of regulatory networks that impact disease progression. For this purpose, we have developed a computational framework, named PRADA, to identify RNA-binding proteins that underlie pathologic gene expression modulations. Using this approach, we uncovered the RNA-binding protein RBMS1 as a novel suppressor of colon cancer progression. Our findings indicate that silencing RBMS1, which is achieved through epigenetic reprogramming, results in increased metastatic capacity in colon cancer cells. Restoring RBMS1 expression, in turn, blunts metastatic capacity. We have shown that RBMS1 functions as a post-transcriptional regulator of RNA stability by binding and stabilizing ~80 target mRNAs. Importantly, our analyses of colon cancer datasets as well as measurements in clinical samples have shown that RBMS1 silencing is associated with disease progression and poor survival. Our findings establish a previously unknown role for RBMS1 in mammalian gene expression regulation and its role in colon cancer metastasis.

### 1. Create and clean up input file (log-fold change)

In this study, we are starting with Illumina arrays from GSE59857, which compares poorly and highly metastatic colon cancer cell lines.


```python
import sys
import os
import pandas as pd
import re
import numpy as np
import scipy as sp
from collections import defaultdict
from itertools import islice
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```


```python
expfile = 'input/high-vs-low_metastatic_lines_GSE59857.txt'
exp = pd.read_csv(expfile, sep='\t', header=0, index_col=0)
exp.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RefSeq</th>
      <th>CACO2</th>
      <th>COLO201</th>
      <th>LS123</th>
      <th>SW480</th>
      <th>SW1417</th>
      <th>LS174T</th>
      <th>COLO320</th>
      <th>HCT116</th>
      <th>HT29</th>
      <th>WIDR</th>
      <th>LOVO</th>
    </tr>
    <tr>
      <th>Probe</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ILMN_3245919</td>
      <td>XM_001714734</td>
      <td>151.323411</td>
      <td>148.142560</td>
      <td>171.410158</td>
      <td>167.456323</td>
      <td>183.939424</td>
      <td>41923.971670</td>
      <td>146.239302</td>
      <td>177.357721</td>
      <td>176.733719</td>
      <td>164.927448</td>
      <td>158.597969</td>
    </tr>
    <tr>
      <td>ILMN_1672148</td>
      <td>NM_020299</td>
      <td>146.968324</td>
      <td>192.024693</td>
      <td>123.071839</td>
      <td>133.238433</td>
      <td>140.839404</td>
      <td>2105.803564</td>
      <td>124.424634</td>
      <td>302.047197</td>
      <td>2930.576970</td>
      <td>12003.061920</td>
      <td>12739.901670</td>
    </tr>
    <tr>
      <td>ILMN_1685387</td>
      <td>NM_002644</td>
      <td>161.659510</td>
      <td>143.659146</td>
      <td>167.381620</td>
      <td>159.185124</td>
      <td>129.949763</td>
      <td>26836.516690</td>
      <td>143.809830</td>
      <td>144.222191</td>
      <td>153.363798</td>
      <td>207.071610</td>
      <td>427.244406</td>
    </tr>
    <tr>
      <td>ILMN_1720998</td>
      <td>NM_001218</td>
      <td>205.546207</td>
      <td>173.576579</td>
      <td>171.661433</td>
      <td>126.627450</td>
      <td>198.623380</td>
      <td>10232.277200</td>
      <td>281.566402</td>
      <td>318.708726</td>
      <td>3647.612966</td>
      <td>12367.323190</td>
      <td>1912.260020</td>
    </tr>
    <tr>
      <td>ILMN_1666536</td>
      <td>NM_014312</td>
      <td>196.454316</td>
      <td>169.608556</td>
      <td>225.914542</td>
      <td>173.089286</td>
      <td>184.556284</td>
      <td>387.418966</td>
      <td>165.364108</td>
      <td>213.687788</td>
      <td>7966.161694</td>
      <td>20813.031510</td>
      <td>207.675049</td>
    </tr>
  </tbody>
</table>
</div>




```python
from scipy.stats import ttest_ind
#Poorly metastatic: CACO2,COLO201,LS123,SW480,SW1417
#Highly metastatic: LS174T,COLO320,HCT116,HT29,WIDR,LOVO
logFC = pd.DataFrame(np.log2(exp.iloc[:,6:12].mean(axis=1) / exp.iloc[:,1:6].mean(axis=1)), columns=['logFC'])
logFC['pval'] = exp.apply(lambda x: ttest_ind(x[1:6], x[6:12], equal_var=False)[1], axis=1)
logFC.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logFC</th>
      <th>pval</th>
    </tr>
    <tr>
      <th>Probe</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ILMN_3245919</td>
      <td>5.437057</td>
      <td>0.363198</td>
    </tr>
    <tr>
      <td>ILMN_1672148</td>
      <td>5.095663</td>
      <td>0.093406</td>
    </tr>
    <tr>
      <td>ILMN_1685387</td>
      <td>4.932240</td>
      <td>0.357067</td>
    </tr>
    <tr>
      <td>ILMN_1720998</td>
      <td>4.773884</td>
      <td>0.083028</td>
    </tr>
    <tr>
      <td>ILMN_1666536</td>
      <td>4.706519</td>
      <td>0.221115</td>
    </tr>
  </tbody>
</table>
</div>




```python
logFC['RefSeq'] = exp['RefSeq']
logFC.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logFC</th>
      <th>pval</th>
      <th>RefSeq</th>
    </tr>
    <tr>
      <th>Probe</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ILMN_3245919</td>
      <td>5.437057</td>
      <td>0.363198</td>
      <td>XM_001714734</td>
    </tr>
    <tr>
      <td>ILMN_1672148</td>
      <td>5.095663</td>
      <td>0.093406</td>
      <td>NM_020299</td>
    </tr>
    <tr>
      <td>ILMN_1685387</td>
      <td>4.932240</td>
      <td>0.357067</td>
      <td>NM_002644</td>
    </tr>
    <tr>
      <td>ILMN_1720998</td>
      <td>4.773884</td>
      <td>0.083028</td>
      <td>NM_001218</td>
    </tr>
    <tr>
      <td>ILMN_1666536</td>
      <td>4.706519</td>
      <td>0.221115</td>
      <td>NM_014312</td>
    </tr>
  </tbody>
</table>
</div>




```python
logFC_r = logFC.groupby(['RefSeq']).agg(np.mean)
logFC_r.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logFC</th>
      <th>pval</th>
    </tr>
    <tr>
      <th>RefSeq</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NM_000014</td>
      <td>-0.145564</td>
      <td>0.121230</td>
    </tr>
    <tr>
      <td>NM_000015</td>
      <td>0.766022</td>
      <td>0.323496</td>
    </tr>
    <tr>
      <td>NM_000016</td>
      <td>-0.544629</td>
      <td>0.281687</td>
    </tr>
    <tr>
      <td>NM_000017</td>
      <td>1.211667</td>
      <td>0.057775</td>
    </tr>
    <tr>
      <td>NM_000018</td>
      <td>0.413733</td>
      <td>0.294752</td>
    </tr>
  </tbody>
</table>
</div>




```python
logFC_r.to_csv('input/high-vs-low_metastatic_lines_GSE59857_logFC_refseq.txt', sep='\t', index=True, index_label='RefSeq')
```


```python
exp_r = exp
exp_r.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RefSeq</th>
      <th>CACO2</th>
      <th>COLO201</th>
      <th>LS123</th>
      <th>SW480</th>
      <th>SW1417</th>
      <th>LS174T</th>
      <th>COLO320</th>
      <th>HCT116</th>
      <th>HT29</th>
      <th>WIDR</th>
      <th>LOVO</th>
    </tr>
    <tr>
      <th>Probe</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ILMN_3245919</td>
      <td>XM_001714734</td>
      <td>151.323411</td>
      <td>148.142560</td>
      <td>171.410158</td>
      <td>167.456323</td>
      <td>183.939424</td>
      <td>41923.971670</td>
      <td>146.239302</td>
      <td>177.357721</td>
      <td>176.733719</td>
      <td>164.927448</td>
      <td>158.597969</td>
    </tr>
    <tr>
      <td>ILMN_1672148</td>
      <td>NM_020299</td>
      <td>146.968324</td>
      <td>192.024693</td>
      <td>123.071839</td>
      <td>133.238433</td>
      <td>140.839404</td>
      <td>2105.803564</td>
      <td>124.424634</td>
      <td>302.047197</td>
      <td>2930.576970</td>
      <td>12003.061920</td>
      <td>12739.901670</td>
    </tr>
    <tr>
      <td>ILMN_1685387</td>
      <td>NM_002644</td>
      <td>161.659510</td>
      <td>143.659146</td>
      <td>167.381620</td>
      <td>159.185124</td>
      <td>129.949763</td>
      <td>26836.516690</td>
      <td>143.809830</td>
      <td>144.222191</td>
      <td>153.363798</td>
      <td>207.071610</td>
      <td>427.244406</td>
    </tr>
    <tr>
      <td>ILMN_1720998</td>
      <td>NM_001218</td>
      <td>205.546207</td>
      <td>173.576579</td>
      <td>171.661433</td>
      <td>126.627450</td>
      <td>198.623380</td>
      <td>10232.277200</td>
      <td>281.566402</td>
      <td>318.708726</td>
      <td>3647.612966</td>
      <td>12367.323190</td>
      <td>1912.260020</td>
    </tr>
    <tr>
      <td>ILMN_1666536</td>
      <td>NM_014312</td>
      <td>196.454316</td>
      <td>169.608556</td>
      <td>225.914542</td>
      <td>173.089286</td>
      <td>184.556284</td>
      <td>387.418966</td>
      <td>165.364108</td>
      <td>213.687788</td>
      <td>7966.161694</td>
      <td>20813.031510</td>
      <td>207.675049</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Generating an RBP-target matrix

In this section, we will create a binary matrix where the rows are transcripts and the columns are RBPs. If the transcript(i) contains a putative binding site for RBP(j), the element(i,j) will be set to '1', otherwise, it will remain '0'.


```python
#RNA Dynamics file: a 3-column tab-delimited file with RefSeq IDs and changes in RNA dynamics (in this case expression)
#and their associated p-values
RDfile = 'input/high-vs-low_metastatic_lines_GSE59857_logFC_refseq.txt'
```


```python
motifs = pd.read_csv('data/motif_RBP_map.txt', sep='\t', header=0)
motifs.set_index('RBP', inplace=True, drop=False)
motifs.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RBP</th>
      <th>motif</th>
    </tr>
    <tr>
      <th>RBP</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MATR3</td>
      <td>MATR3</td>
      <td>[AC]ATCTT[AG]</td>
    </tr>
    <tr>
      <td>ENOX1</td>
      <td>ENOX1</td>
      <td>[ACT][AG][TG]ACAG</td>
    </tr>
    <tr>
      <td>PTBP1</td>
      <td>PTBP1</td>
      <td>[ACT][TC]TTT[TC]T</td>
    </tr>
    <tr>
      <td>RBMS3</td>
      <td>RBMS3</td>
      <td>[ACT]ATATA</td>
    </tr>
    <tr>
      <td>RBM6</td>
      <td>RBM6</td>
      <td>[ACT]ATCCA[AG]</td>
    </tr>
  </tbody>
</table>
</div>




```python
#write motifs to file
motifs['motif'].unique().tofile('outputs/motifs.txt', sep="\n", format="%s")
```


```python
#match motifs to fasta file
#needs genregexp
import subprocess
cmd = "gunzip data/hg19_mrna.fa.gz"
print(cmd)
subprocess.call(cmd,shell=True)
cmd = "perl programs/scan_fasta_for_regex_matches.pl outputs/motifs.txt data/hg19_mrna.fa 1 > outputs/motifs-mrna.out"
print(cmd)
subprocess.call(cmd,shell=True)
```

    gunzip data/hg19_mrna.fa.gz
    perl programs/scan_fasta_for_regex_matches.pl outputs/motifs.txt data/hg19_mrna.fa 1 > outputs/motifs-mrna.out





    0




```python
rmap = defaultdict(dict)
motif=""
refs = {}
with open("outputs/motifs-mrna.out", "rt") as f:
    for l in f:
        if l.startswith('>'):
            l=re.sub('\s+$','',l)
            motif = l[1:]
            continue
        a = l.split('\t')
        rmap[motif][a[0]]=1
        refs[a[0]] = 1

```


```python
exp = pd.read_csv(RDfile, sep='\t', header=0, index_col=0)
exp.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logFC</th>
      <th>pval</th>
    </tr>
    <tr>
      <th>RefSeq</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NM_000014</td>
      <td>-0.145564</td>
      <td>0.121230</td>
    </tr>
    <tr>
      <td>NM_000015</td>
      <td>0.766022</td>
      <td>0.323496</td>
    </tr>
    <tr>
      <td>NM_000016</td>
      <td>-0.544629</td>
      <td>0.281687</td>
    </tr>
    <tr>
      <td>NM_000017</td>
      <td>1.211667</td>
      <td>0.057775</td>
    </tr>
    <tr>
      <td>NM_000018</td>
      <td>0.413733</td>
      <td>0.294752</td>
    </tr>
  </tbody>
</table>
</div>




```python
mat = pd.DataFrame(0, index=list(set(exp.index) & set(refs.keys())), columns=motifs['RBP'])
```


```python
for rbp in mat.columns:
    m = motifs.loc[rbp,'motif']
    #print(m)
    #sys.stdout.flush()
    ref = rmap[m].keys()
    ref = list(set(rmap[m].keys()) & set(mat.index))
    mat.loc[ref,rbp] = 1
mat.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>RBP</th>
      <th>MATR3</th>
      <th>ENOX1</th>
      <th>PTBP1</th>
      <th>RBMS3</th>
      <th>RBM6</th>
      <th>LIN28A</th>
      <th>HNRNPC</th>
      <th>HNRNPCL1</th>
      <th>SNRNP70</th>
      <th>RBM8A</th>
      <th>...</th>
      <th>SRSF2</th>
      <th>HNRNPH2</th>
      <th>DAZAP1</th>
      <th>MSI1</th>
      <th>ESRP2</th>
      <th>ZC3H14</th>
      <th>TIA1</th>
      <th>U2AF2</th>
      <th>CPEB4</th>
      <th>RALY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NM_173362</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>NM_006196</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>NM_016579</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>NM_001700</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>NM_001920</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 75 columns</p>
</div>




```python
mat.to_csv('outputs/RBP-v-RefSeq_target_matrix.txt', sep='\t', index=True, index_label='RefSeq')
```

### 3. Generating the proper matrices

The general form of the model:

<img src="https://render.githubusercontent.com/render/math?math=\Delta Exp(g)=\sum_{i} \alpha_{i} \cdot t_{i,g} \cdot \Delta Exp(RBP_{i}) + c_{g}">

Contraints:

<img src="https://render.githubusercontent.com/render/math?math=min\ \frac{1}{2n} \left [ \left \| \alpha X - Exp \right \|+ \lambda \sum_{i}\frac{\left | \alpha_{i} \right |}{\left | \Delta Exp\left ( RBP_{i} \right )\right |} \right]">


```python
hgnc_to_ref = pd.read_csv('data/hg19_genes_vs_refseq.txt', sep='\t', header=0)
hgnc_to_ref.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HGNC</th>
      <th>RefSeq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1/2-SBSRNA4</td>
      <td>NR_039978</td>
    </tr>
    <tr>
      <td>1</td>
      <td>A1BG</td>
      <td>NM_130786</td>
    </tr>
    <tr>
      <td>2</td>
      <td>A1BG-AS1</td>
      <td>NR_015380</td>
    </tr>
    <tr>
      <td>3</td>
      <td>A1CF</td>
      <td>NM_001198818</td>
    </tr>
    <tr>
      <td>4</td>
      <td>A1CF</td>
      <td>NM_001198819</td>
    </tr>
  </tbody>
</table>
</div>




```python
hgnc_to_ref = hgnc_to_ref.groupby('HGNC')['RefSeq'].apply(lambda x: "%s" % ','.join(x))
```


```python
mat = pd.read_csv('outputs/RBP-v-RefSeq_target_matrix.txt', sep='\t', header=0, index_col=0)
mat.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MATR3</th>
      <th>ENOX1</th>
      <th>PTBP1</th>
      <th>RBMS3</th>
      <th>RBM6</th>
      <th>LIN28A</th>
      <th>HNRNPC</th>
      <th>HNRNPCL1</th>
      <th>SNRNP70</th>
      <th>RBM8A</th>
      <th>...</th>
      <th>SRSF2</th>
      <th>HNRNPH2</th>
      <th>DAZAP1</th>
      <th>MSI1</th>
      <th>ESRP2</th>
      <th>ZC3H14</th>
      <th>TIA1</th>
      <th>U2AF2</th>
      <th>CPEB4</th>
      <th>RALY</th>
    </tr>
    <tr>
      <th>RefSeq</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NM_173362</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>NM_006196</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>NM_016579</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>NM_001700</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>NM_001920</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 75 columns</p>
</div>




```python
for rbp in mat.columns:
    #print(rbp)
    rbp_refs = hgnc_to_ref[rbp].split(',')
    rbp_sum = 0
    rbp_cnt = 0
    rbp_max = 0
    for r in rbp_refs:
        if r in exp.index:
            if (abs(exp.loc[r,'logFC']) > rbp_max):
                motifs.loc[rbp,'diff'] = exp.loc[r,'logFC']
                motifs.loc[rbp,'pval'] = exp.loc[r,'pval']
                rbp_max = abs(exp.loc[r,'logFC'])

motifs.to_csv('outputs/RBP_motif_diff.txt', sep='\t', index=True, index_label='RefSeq')
```


```python
for rbp in mat.columns:
    mat[rbp] = mat[rbp]*motifs.loc[rbp,'diff']
mat.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MATR3</th>
      <th>ENOX1</th>
      <th>PTBP1</th>
      <th>RBMS3</th>
      <th>RBM6</th>
      <th>LIN28A</th>
      <th>HNRNPC</th>
      <th>HNRNPCL1</th>
      <th>SNRNP70</th>
      <th>RBM8A</th>
      <th>...</th>
      <th>SRSF2</th>
      <th>HNRNPH2</th>
      <th>DAZAP1</th>
      <th>MSI1</th>
      <th>ESRP2</th>
      <th>ZC3H14</th>
      <th>TIA1</th>
      <th>U2AF2</th>
      <th>CPEB4</th>
      <th>RALY</th>
    </tr>
    <tr>
      <th>RefSeq</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NM_173362</td>
      <td>-1.129317</td>
      <td>0.05266</td>
      <td>0.229692</td>
      <td>0.079549</td>
      <td>-0.000000</td>
      <td>-0.055535</td>
      <td>0.79533</td>
      <td>0.161332</td>
      <td>0.462524</td>
      <td>-0.017537</td>
      <td>...</td>
      <td>-0.638921</td>
      <td>-0.0</td>
      <td>0.368176</td>
      <td>0.623895</td>
      <td>0.000000</td>
      <td>0.120822</td>
      <td>-1.240809</td>
      <td>0.0</td>
      <td>-1.366578</td>
      <td>-0.146491</td>
    </tr>
    <tr>
      <td>NM_006196</td>
      <td>-0.000000</td>
      <td>0.00000</td>
      <td>0.229692</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.79533</td>
      <td>0.161332</td>
      <td>0.462524</td>
      <td>-0.017537</td>
      <td>...</td>
      <td>-0.638921</td>
      <td>-0.0</td>
      <td>0.368176</td>
      <td>0.000000</td>
      <td>0.027613</td>
      <td>0.120822</td>
      <td>-1.240809</td>
      <td>0.0</td>
      <td>-1.366578</td>
      <td>-0.146491</td>
    </tr>
    <tr>
      <td>NM_016579</td>
      <td>-0.000000</td>
      <td>0.05266</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.017537</td>
      <td>...</td>
      <td>-0.638921</td>
      <td>-0.0</td>
      <td>0.368176</td>
      <td>0.000000</td>
      <td>0.027613</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.0</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <td>NM_001700</td>
      <td>-0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.462524</td>
      <td>-0.000000</td>
      <td>...</td>
      <td>-0.638921</td>
      <td>-0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.0</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <td>NM_001920</td>
      <td>-1.129317</td>
      <td>0.05266</td>
      <td>0.229692</td>
      <td>0.079549</td>
      <td>-0.950022</td>
      <td>-0.055535</td>
      <td>0.79533</td>
      <td>0.161332</td>
      <td>0.462524</td>
      <td>-0.017537</td>
      <td>...</td>
      <td>-0.638921</td>
      <td>-0.0</td>
      <td>0.368176</td>
      <td>0.623895</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.0</td>
      <td>-1.366578</td>
      <td>-0.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 75 columns</p>
</div>




```python
mat.to_csv('outputs/RBP-v-RefSeq_target_matrix_dExp.txt', sep='\t', index=True, index_label='RefSeq')
```


```python
#penalties are defined as 1/|dExp|
penalties = pd.DataFrame(index=motifs.index)
penalties['penalties'] = motifs['diff'].apply(lambda x: 1/abs(x))
penalties.to_csv('outputs/penalties.txt', sep='\t', index=True, index_label='RBP')
```


```python
exp_fil = pd.DataFrame(index=mat.index)
exp_fil['logFC'] = exp.loc[mat.index,'logFC']
exp_fil.to_csv('outputs/high-vs-low_metastatic_lines_GSE59857_logFC_refseq_fil.txt', sep='\t', index=True, index_label='RefSeq')
```

### 4. Deriving coefficients

Here, we use gneralized linear models (lasso regression with custom penalty terms) to identify RBPs whose expression is informative for predicting the expression of their putative regulon. This custom penalty term ensures that RBPs whose activity does not change are not selected by the model. This also stabilizes the resulting model, which would otherwise be a major issue as RBPs that belong to the same family often have very similar binding preferences resulting in correlated features in the interaction matrix. After running this regression analysis, the RBPs with the largest assigned coefficients (absolute value) are prioritized.


```python
import rpy2
%load_ext rpy2.ipython
```


```python
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
```


```r
%%R -o coef,fit,x,y,y.t
library(Matrix)
library(glmnet)
library(tidyverse)
x <- read.table('outputs/RBP-v-RefSeq_target_matrix_dExp.txt', row.names=1, header=TRUE, sep="\t")
y <- read.table('outputs/high-vs-low_metastatic_lines_GSE59857_logFC_refseq_fil.txt', row.names=1, header=TRUE, sep="\t")
p.fac <- read.table('outputs/penalties.txt', row.names=1, header=TRUE, sep="\t")

library(bestNormalize)
(BNobject <- bestNormalize(as.matrix(y), quiet=T))

fit <- glmnet(as.matrix(x), BNobject$x.t, penalty.factor=as.numeric(unlist(p.fac)), family="gaussian", alpha=1)
y.t <- BNobject$x.t

coef <- as.matrix(coef(fit))
colnames(coef) <- apply(abs(fit$beta), 2, sum) #set L1 norm as the header
coef <- coef[,round(as.numeric(colnames(coef)), digits=2)<=0.75]
coef <- coef[rowSums(coef)!=0,]
coef <- coef[order(coef[,dim(coef)[2]], decreasing=T),]
coef <- coef[-1,]
x <- as_tibble(x, rownames="RefSeq")
y <- as_tibble(y, rownames="RefSeq")
fit$dev
write.table(coef, file='outputs/high-vs-low_metastatic_lines_GSE59857_coef.txt',sep="\t", row.names=TRUE, col.names=NA, quote=F)
```


```r
%%R -i coef
library(tidyverse)
library(reshape2)
df <- as_tibble(melt(coef))
colnames(df) <- c('RBP', 'norm', 'coefficient')
df %>% ggplot(aes(x=norm, y=coefficient, group=RBP, color=RBP)) + geom_line(size=1.5) + theme_bw(12) + 
xlab("L1 Norm") + ylab("Coefficient") +
theme(text = element_text(size=20), panel.grid.minor = element_blank(), panel.grid.major = element_blank())
#ggsave("outputs/PRADA_coeffcients_vs_L1norm.pdf", width=4.5, height=3)
```


![png](PRADA_files/PRADA_41_0.png)



```python

```
