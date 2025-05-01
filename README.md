# Advances and Challenges in Data×LLM

* [0. System & Review](#0-system--review)
* [1. Data Processing for LLM](#1-data-processing-for-llm)
    * [1.1 Data Acquisition for LLM](#11-data-acquisition-for-llm)    
    * [1.2 Data Deduplication for LLM](#12-data-deduplication-for-llm)    
    * [1.3 Data Filtering for LLM](#13-data-filtering-for-llm)    
    * [1.4 Data Transformation for LLM](#14-data-transformation-for-llm)    
    * [1.5 Data Selection for LLM](#15-data-selection-for-llm)    
    * [1.6 Data Mixing for LLM](#16-data-mixing-for-llm)    
    * [1.7 Data Synthesis and Augmentation for LLM](#17-data-synthesis-and-augmentation-for-llm)
    * [1.8 Data Processing Pipelines for LLM](#18-data-processing-pipelines-for-llm)
    * [1.9.1 Data Provenance for LLM](#191-data-provenance-for-llm)
    * [1.9.2 Data Visualization for LLM](#192-data-visualization-for-llm)
    * [1.9.3 Constructing Dense LLMs](#193-constructing-dense-llms)
* [2. Data Storage for LLM](#2-data-storage-for-llm)
    * [2.1 Data Storage for Training](#21-data-storage-for-training)
    * [2.2 Data Storage for Inference](#22-data-storage-for-inference)
    * [2.3 Data Storage for RAG](#23-data-storage-for-rag)
* [3. Data Serving for LLM](#3-data-serving-for-llm)
    * [3.1 Data Serving for Training](#31-data-serving-for-training)
    * [3.2 Data Serving for Inference](#32-data-serving-for-inference)
    * [3.3 Data Serving for RAG](#33-data-serving-for-rag)
* [4. LLM for Data Processing](#4-llm-for-data-processing)
    * [4.1 Data Cleaning](#41-data-cleaning)
    * [4.2 Entity Matching](#42-entity-matching)
    * [4.3 Schema Matching](#43-schema-matching)
    * [4.4 Data Discovery](#44-data-discovery)
* [5. LLM for Data Analysis](#5-llm-for-data-analysis)
    * [5.1 Structured Data Analysis](#51-structured-data-analysis)
    * [5.2 Semi-Structured Data Analysis](#52-semi-structured-data-analysis)
    * [5.3 Unstructured Data Analysis](#53-unstructured-data-analysis)
    * [5.4 Data Exploration](#54-data-exploration)
    * [5.5 Data Visualization](#55-data-visualization)
* [6. LLM for Data System Optimization](#6-llm-for-data-system-optimization)
    * [6.1 Configuration Tuning](#61-configuration-tuning)
    * [6.2 Query Optimization](#62-query-optimization)
    * [6.3 Anomaly  Diagnosis](#63-anomaly-diagnosis)


## 0. System & Review

**Scaling Laws for Data Filtering -- Data Curation cannot be Compute Agnostic**  
Sachin Goyal, Pratyush Maini, Zachary C. Lipton, Aditi Raghunathan, J. Zico Kolter. *CVPR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2404.07177 )] 

**Data-efficient Fine-tuning for LLM-based Recommendation**  
Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, Tat-Seng Chua. *SIGIR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.17197 )] 

**The Synergy between Data and Multi-Modal Large Language Models: A Survey from Co-Development Perspective**  
Zhen Qin, Daoyuan Chen, Wenhao Zhang, Liuyi Yao, Yilun Huang, Bolin Ding, Yaliang Li, Shuiguang Deng. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2407.08583 )]

**A Survey on Data Selection for Language Models**  
Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, Colin Raffel, Shiyu Chang, Tatsunori Hashimoto, William Yang Wang. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.16827 )] 

**Survey of vector database management systems**  
James Jie Pan, Jianguo Wang, Guoliang Li. *VLDB 2024.* [[pdf](https://doi.org/10.1007/s00778-024-00864-x )] 

**Large Language Model for Table Processing: A Survey**  
Weizheng Lu, Jing Zhang, Ju Fan, Zihao Fu, Yueguo Chen, Xiaoyong Du. *FCS 2024.* [[pdf](https://arxiv.org/abs/2402.05121 )] 

**Large Language Models for Data Annotation and Synthesis: A Survey**  
Zhen Tan, Dawei Li, Song Wang, Alimohammad Beigi, Bohan Jiang, Amrita Bhattacharjee, Mansooreh Karami, Jundong Li, Lu Cheng, Huan Liu. *EMNLP 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.13446 )] 

**On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey**  
Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, Haobo Wang. *ACL 2024.* [[pdf](https://aclanthology.org/2024.findings-acl.658/ )] 

**A survey on multimodal large language models**  
Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, Enhong Chen. *National Science Review 2024.* [[pdf](https://arxiv.org/abs/2306.13549 )]

**When Large Language Models Meet Vector Databases: A Survey**

*Zhi Jing, Yongye Su, Yikun Han, Bo Yuan, Haiyun Xu, Chunjiang Liu, Kehai Chen, Min Zhang. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2402.01763)]

**LLM-Enhanced Data Management**

*Xuanhe Zhou, Xinyang Zhao, Guoliang Li. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2402.02643)]

**Trustworthy and Efficient LLMs Meet Databases**

*Kyoungmin Kim, Anastasia Ailamaki. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2412.18022)]

**Applications and Challenges for Large Language Models: From Data Management Perspective**

*Zhang, Meihui, Zhaoxuan Ji, Zhaojing Luo, Yuncheng Wu, and Chengliang Chai. ICDE 2024.* [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10598077)]

**Demystifying Data Management for Large Language Models**

*Xupeng Miao, Zhihao Jia, and Bin Cui. SIGMOD 2024.* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3626246.3654683)]

**A Survey on Data Selection for LLM Instruction Tuning**
 Jiahao Wang, Bolin Zhang, Qianlong Du, Jiajun Zhang, Dianhui Chu. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2402.05123)]

**Graph Learning in the Era of LLMs: A Survey from the Perspective of Data, Models, and Tasks** 

Xunkai Li, Zhengyu Wu, Jiayi Wu, Hanwen Cui, Jishuo Jia, Rong-Hua Li, Guoren Wang. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2412.12456)]

**Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey** 

Qizhi Pei, Lijun Wu, Kaiyuan Gao, Jinhua Zhu, Yue Wang, Zun Wang, Tao Qin, Rui Yan. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.01528)]

**DB-GPT: Large Language Model Meets Database**

*Xuanhe Zhou, Zhaoyan Sun, Guoliang Li. Data Science and Engineering 2023.* [[pdf](http://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbgpt-dse.pdf)]

**When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale**  
Max Marion, Ahmet Üstün, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, Sara Hooker. *NeurIPS 2023.* [[pdf](https://doi.org/10.48550/arXiv.2309.04564 )] 

**How Large Language Models Will Disrupt Data Management**

*Raul Castro Fernandez, Aaron J. Elmore, Michael J. Franklin, Sanjay Krishnan, Chenhao Tan. VLDB 2023.* [[pdf](https://www.vldb.org/pvldb/vol16/p3302-fernandez.pdf)]

**From Large Language Models to Databases and Back: A Discussion on Research and Education**

*Sihem Amer-Yahia, Angela Bonifati, Lei Chen, Guoliang Li, Kyuseok Shim, Jianliang Xu, Xiaochun Yang. SIGMOD Record 2023.* [[pdf](https://sigmodrecord.org/publications/sigmodRecord/2309/pdfs/09_OpenForum_AmerYahia.pdf)]

**Data Management For Training Large Language Models: A Survey**  
Zige Wang, Wanjun Zhong, Yufei Wang, Qi Zhu, Fei Mi, Baojun Wang, Lifeng Shang, Xin Jiang, Qun Liu. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2312.01700 )]

**Survey of Hallucination in Natural Language Generation**
 Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, Pascale Fung. *ACM Computing Surveys 2023.* [[pdf](https://doi.org/10.1145/3571730)]

**Retrieval-Augmented Generation for Large Language Models: A Survey**
 Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2312.10997)]

**Can Foundation Models Wrangle Your Data?**

*Avanika Narayan, Ines Chami, Laurel J. Orr, Christopher Ré. VLDB 2022.* [[pdf](https://www.vldb.org/pvldb/vol16/p738-narayan.pdf)]

**A comprehensive survey on data provenance: State-of-the-art approaches and their deployments for IoT security enforcement**
 Md Morshed Alam, Weichao Wang. *Journal of Computer Security 2021.* [[pdf](https://doi.org/10.3233/JCS-200108)]

**Big data storage technologies: a survey**  
Aisha Siddiqa, Ahmad Karim, Abdullah Gani. *Frontiers of Information Technology & Electronic Engineering 2017.* [[pdf](https://link.springer.com/article/10.1631/FITEE.1500441 )]

**Survey of Graph Database Models** 

Renzo Angles, Claudio Gutierrez. *CSUR 2008.* [[pdf](https://doi.org/10.1145/1322432.1322433)]




## 1. Data Processing for LLM

**DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**  
DeepSeek-AI, Daya Guo, Dejian Yang, et al. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.12948 )]

**RedPajama: An Open Dataset for Training Large Language Models**  
Maurice Weber, Daniel Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun, Rahul Chalamala, Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, Ce Zhang. *NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2411.12372 )]

**The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale**  
Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf. *NeurIPS 2024.* [[pdf](https://proceedings.neurips.cc/paper_files/paper/2024/hash/370df50ccfdf8bde18f8f9c2d9151bda-Abstract-Datasets_and_Benchmarks_Track.html )]

**MM-LLMs: Recent Advances in MultiModal Large Language Models**  
Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui Chu, Dong Yu. *ACL 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.13601 )]

**Continual pre-training mitigates forgetting in language and vision**  
Andrea Cossu, Antonio Carta, Lucia Passaro, Vincenzo Lomonaco, Tinne Tuytelaars, Davide Bacciu. *Neural Networks 2024.* [[pdf](https://doi.org/10.1016/j.neunet.2024.106492 )]

**UltraFeedback: Boosting Language Models with Scaled AI Feedback**  
Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, Maosong Sun. *ICML 2024.* [[pdf](https://doi.org/10.48550/arXiv.2310.01377 )]

**Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval - Augmented Generation**  
Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, Vicente Grau. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2408.04187 )]

**LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models**  
Haitao Li, You Chen, Qingyao Ai, Yueyue Wu, Ruizhe Zhang, Yiqun Liu. *NeurIPS 2024.* [[pdf](https://papers.nips.cc/paper_files/paper/2024/hash/2cb40fc022ca7bdc1a9a78b793661284-Abstract-Datasets_and_Benchmarks_Track.html )]

**MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI**  
Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen. *CVPR 2024.* [[pdf](https://ieeexplore.ieee.org/abstract/document/10656299 )]

**OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents**  
Hugo Laurençon, Lucile Saulnier, Leo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, Matthieu Cord, Victor Sanh. *NeurIPS 2023.* [[pdf](https://papers.nips.cc/paper_files/paper/2023/hash/e2cfb719f58585f779d0a4f9f07bd618-Abstract-Datasets_and_Benchmarks.html )]

**BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark** 

Dakuan Lu, Hengkui Wu, Jiqing Liang, Yipei Xu, Qianyu He, Yipeng Geng, Mengkun Han, Yingsi Xin, Yanghua Xiao. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2302.09432 )]

**DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services**  
Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, Zhongyu Wei. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2309.11325 )]

**Evaluating Large Language Models Trained on Code**  
Mark Chen, Jerry Tworek, Heewoo Jun, et al. *arXiv 2021.* [[pdf](https://arxiv.org/abs/2107.03374v2 )]

**What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams**  
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, Peter Szolovits. *AAAI 2021.* [[pdf](https://arxiv.org/abs/2009.13081v1 )]

**Advances in natural language processing**  
Julia Hirschberg, Christopher D. Manning. *Science 2015.* [[pdf](https://doi.org/10.1126/science.aaa8685 )]

### 1.1 Data Acquisition for LLM

**CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages**
 Thuat Nguyen, Chien Van Nguyen, Viet Dac Lai, Hieu Man, Nghia Trung Ngo, Franck Dernoncourt, Ryan A. Rossi, Thien Huu Nguyen. *LREC-COLING 2024.* [[pdf](https://aclanthology.org/2024.lrec-main.377.pdf)]

**General OCR Theory: Towards OCR - 2.0 via a Unified End - to - end Model**
 Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, Chunrui Han, Xiangyu Zhang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2409.01704v1)]

**MinerU: An Open-Source Solution for Precise Document Content Extraction**
 Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, Conghui He. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2409.18839)]

**YOLOv10: Real-Time End-to-End Object Detection**
 Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding. *NeurIPS 2024.* [[pdf](https://arxiv.org/abs/2405.14458)]

**UMIE: Unified Multimodal Information Extraction with Instruction Tuning**
 Lin Sun, Kai Zhang, Qingyuan Li, Renze Lou. *AAAI 2024.* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/29873)]

**Focus Anywhere for Fine-grained Multi-page Document Understanding**
 Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, Xiangyu Zhang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2405.14295)]

**Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models**
 Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, Xiangyu Zhang. *ECCV 2024.* [[pdf](https://link.springer.com/chapter/10.1007/978-3-031-73235-5_23)]

**WebIE: Faithful and Robust Information Extraction on the Web**
 Chenxi Whitehouse, Clara Vania, Alham Fikri Aji, Christos Christodoulopoulos, Andrea Pierleoni. *ACL 2023.* [[pdf](https://aclanthology.org/2023.acl-long.428/)]

**An Empirical Comparison of Web Content Extraction Algorithms**
 Janek Bevendorff, Sanket Gupta, Johannes Kiesel, Benno Stein. *SIGIR 2023.* [[pdf](https://doi.org/10.1145/3539618.3591920)]

**ReFinED: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking**
 Tom Ayoola, Shubhi Tyagi, Joseph Fisher, Christos Christodoulopoulos, Andrea Pierleoni. *NAACL 2022.* [[pdf](https://aclanthology.org/2022.naacl-industry.24.pdf)]

**Alignment-Augmented Consistent Translation for Multilingual Open Information Extraction**
 Keshav Kolluru, Muqeeth Mohammed, Shubham Mittal, Soumen Chakrabarti, Mausam. *ACL 2022.* [[pdf](https://aclanthology.org/2022.acl-long.179/)]

**Optimizing Data Collection for Machine Learning**
 Rafid Mahmood, James Lucas, Jose M. Alvarez, Sanja Fidler, Marc T. Law. *NeurIPS 2022.* [[pdf](https://arxiv.org/abs/2210.01234)]

**LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking**
 Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei. *ACM Multimedia 2022.* [[pdf](https://arxiv.org/abs/2204.08387)]

**The Stack: 3 TB of permissively licensed source code**
 Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, Harm de Vries. *arXiv 2022.* [[pdf](https://arxiv.org/abs/2211.15533v1)]

**mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer**
 Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel. *NAACL 2021.* [[pdf](https://aclanthology.org/2021.naacl-main.41.pdf)]

**Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction**
 Adrien Barbaresi. *ACL-IJCNLP 2021.* [[pdf](https://aclanthology.org/2021.acl-demo.15.pdf)]

**Learning Transferable Visual Models From Natural Language Supervision**
 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. *ICML 2021.* [[pdf](https://proceedings.mlr.press/v139/radford21a)]

**Exploring the limits of transfer learning with a unified text-to-text transformer**
 Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. *JMLR 2020.* [[pdf](https://arxiv.org/abs/1910.10683)]

**CodeSearchNet Challenge: Evaluating the State of Semantic Code Search**
 Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, Marc Brockschmidt. *arXiv 2019.* [[pdf](https://arxiv.org/abs/1909.09436v3)]

**Aligning Books and Movies: Towards Story-Like Visual Explanations by Watching Movies and Reading Books**
 Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, Sanja Fidler. *ICCV 2015.* [[pdf](https://doi.org/10.1109/ICCV.2015.11)]

**Content extraction using diverse feature sets**
 Matthew E. Peters, Dan Lecocq. *WWW 2013.* [[pdf](https://doi.org/10.1145/2487788.2487828)]

**Tesseract: An Open-Source Optical Character Recognition Engine**
 Anthony Kay. *Linux Journal 2007.* [[pdf](https://dl.acm.org/doi/10.5555/1288165.1288167)]

**Fact or Fiction: Content Classification for Digital Libraries**
 Aidan Finn, N. Kushmerick, Barry Smyth. *DELOS Workshops / Conferences 2001.* [[pdf](https://www.semanticscholar.org/paper/Fact-or-Fiction%3A-Content-Classification-for-Digital-Finn-Kushmerick/73ccd5c477b37a082f66557a1793852d405e4b6d)]


### 1.2 Data Deduplication for LLM

**Data-Juicer: A One-Stop Data Processing System for Large Language Models**
 Daoyuan Chen, Yilun Huang, Zhijian Ma, Hesen Chen, Xuchen Pan, Ce Ge, Dawei Gao, Yuexiang Xie, Zhaoyang Liu, Jinyang Gao, Yaliang Li, Bolin Ding, Jingren Zhou. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3626246.3653385)]

**LSHBloom: Memory-efficient, Extreme-scale Document Deduplication**
 Arham Khan, Robert Underwood, Carlo Siebenschuh, Yadu Babuji, Aswathy Ajith, Kyle Hippe, Ozan Gokdemir, Alexander Brace, Kyle Chard, Ian Foster. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2411.04257)]

**Effective Pruning of Web-Scale Datasets Based on Complexity of Concept Clusters**
 Amro Abbas, Evgenia Rusak, Kushal Tirumala, Wieland Brendel, Kamalika Chaudhuri, Ari S. Morcos. *ICLR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.04578)]

**FairDeDup: Detecting and Mitigating Vision-Language Fairness Disparities in Semantic Dataset Deduplication**
 Eric Slyman, Stefan Lee, Scott Cohen, Kushal Kafle. *CVPR 2024.* [[pdf](https://arxiv.org/abs/2404.16123)]

**SoftDedup: an Efficient Data Reweighting Method for Speeding Up Language Model Pre-training**
 Nan He, Weichen Xiong, Hanwen Liu, Yi Liao, Lei Ding, Kai Zhang, Guohua Tang, Xiao Han, Yang Wei. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.220/)]

**BaichuanSEED: Sharing the Potential of ExtensivE Data Collection and Deduplication by Introducing a Competitive Large Language Model Baseline**
 Guosheng Dong, Da Pan, Yiding Sun, Shusen Zhang, Zheng Liang, Xin Wu, Yanjun Shen, Fan Yang, Haoze Sun, Tianpeng Li, Mingan Lin, Jianhua Xu, Yufan Zhang, Xiaonan Nie, Lei Su, Bingning Wang, Wentao Zhang, Jiaxin Mao, Zenan Zhou, Weipeng Chen. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2408.15079)]

**Analysis of the Reasoning with Redundant Information Provided Ability of Large Language Models**
 Wenbei Xie. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2310.04039v1)]

**SlimPajama-DC: Understanding Data Combinations for LLM Training**
 Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Zhengzhong Liu, Hongyi Wang, Bowen Tan, Joel Hestness, Natalia Vassilieva, Daria Soboleva, Eric Xing. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2309.10818)]

**SemDeDup: Data-efficient learning at web-scale through semantic deduplication**
 Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, Ari S. Morcos. *ICLR 2023.* [[pdf](https://iclr.cc/virtual/2023/13610)]

**D4: Improving LLM Pretraining via Document De-Duplication and Diversification**
 Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari Morcos. *NeurIPS 2023.* [[pdf](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a8f8cbd7f7a5fb2c837e578c75e5b615-Abstract-Datasets_and_Benchmarks.html)]

**DotHash: Estimating Set Similarity Metrics for Link Prediction and Document Deduplication**
 Igor Nunes, Mike Heddes, Pere Vergés, Danny Abraham, Alex Veidenbaum, Alex Nicolau, Tony Givargis. *KDD 2023.* [[pdf](https://doi.org/10.1145/3580305.3599314)]

**Noise-Robust De-Duplication at Scale**
 Emily Silcock, Luca D'Amico-Wong, Jinglin Yang, Melissa Dell. *arXiv 2022.* [[pdf](https://arxiv.org/abs/2210.04261)]

**Deduplicating Training Data Mitigates Privacy Risks in Language Models**
 Nikhil Kandpal, Eric Wallace, Colin Raffel. *ICML 2022.* [[pdf](https://arxiv.org/abs/2202.06539v3)]

**Scaling Laws and Interpretability of Learning from Repeated Data**
 Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Tom Henighan, Tristan Hume, Scott Johnston, Ben Mann, Chris Olah, Catherine Olsson, Dario Amodei, Nicholas Joseph, Jared Kaplan, Sam McCandlish. *arXiv 2022.* [[pdf](https://arxiv.org/abs/2205.10487)]

**Deduplicating Training Data Makes Language Models Better**
 Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini. *ACL 2022.* [[pdf](https://arxiv.org/abs/2107.06499)]

**Noise-Robust De-Duplication at Scale**
 Emily Silcock, Luca D'Amico-Wong, Jinglin Yang, Melissa Dell. *arXiv 2022.* [[pdf](https://arxiv.org/abs/2210.04261v2)]

**OPT: Open Pre-trained Transformer Language Models**
 Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, Luke Zettlemoyer. *arXiv 2022.* [[pdf](https://arxiv.org/abs/2205.01068v4)]

**LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs**
 Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, Aran Komatsuzaki. *NeurIPS 2021.* [[pdf](https://doi.org/10.48550/arXiv.2111.02114)]

**Learning Transferable Visual Models From Natural Language Supervision**
 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. *ICML 2021.* [[pdf](https://proceedings.mlr.press/v139/radford21a)]

**In Defense of Minhash over Simhash**
 Anshumali Shrivastava, Ping Li. *AISTATS 2014.* [[pdf](https://proceedings.mlr.press/v33/shrivastava14.html)]

**Similarity estimation techniques from rounding algorithms**
 Moses S. Charikar. *STOC 2002.* [[pdf](https://doi.org/10.1145/509907.509965)]

**On the Resemblance and Containment of Documents**
 A. Broder. *Compression and Complexity of SEQUENCES 1997.* [[pdf](https://doi.org/10.1109/SEQUEN.1997.666900)]

**Suffix arrays: a new method for on-line string searches**
 Udi Manber, Gene Myers. *SIAM Journal on Computing 1993.* [[pdf](https://doi.org/10.1137/0222058)]

### 1.3 Data Filtering for LLM

**Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models**
 Zachary Ankner, Cody Blakeney, Kartik Sreenivasan, Max Marion, Matthew L. Leavitt, Mansheej Paul. *ICLR 2025.* [[pdf](https://iclr.cc/virtual/2025/poster/31214)]

**SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection**
 Han Shen, Pin-Yu Chen, Payel Das, Tianyi Chen. *ICLR 2025.* [[pdf](https://iclr.cc/virtual/2025/poster/29422)]

**Data-efficient Fine-tuning for LLM-based Recommendation**
 Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, Tat-Seng Chua. *SIGIR 2024.* [[pdf](https://arxiv.org/abs/2401.17197)]

**Effective Pruning of Web-Scale Datasets Based on Complexity of Concept Clusters**
 Amro Abbas, Evgenia Rusak, Kushal Tirumala, Wieland Brendel, Kamalika Chaudhuri, Ari S. Morcos. *ICLR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.04578)]

**Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning**
 Ming Li, Yong Zhang, Shwai He, Zhitao Li, Hongyu Zhao, Jianzong Wang, Ning Cheng, Tianyi Zhou. *ACL 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.00530)]

**SHED: Shapley-Based Automated Dataset Refinement for Instruction Fine-Tuning**
 Yexiao He, Ziyao Wang, Zheyu Shen, Guoheng Sun, Yucong Dai, Yongkai Wu, Hongyi Wang, Ang Li. *NeurIPS 2024.* [[pdf](https://arxiv.org/abs/2405.00705)]

**QuRating: Selecting High-Quality Data for Training Language Models**
 Alexander Wettig, Aatmik Gupta, Saumya Malik, Danqi Chen. *ICML 2024.* [[pdf](https://arxiv.org/abs/2402.09739)]

**What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning**
 Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He. *ICLR 2024.* [[pdf](https://arxiv.org/abs/2312.15685)]

**Scaling Laws for Data Filtering -- Data Curation cannot be Compute Agnostic**
 Sachin Goyal, Pratyush Maini, Zachary C. Lipton, Aditi Raghunathan, J. Zico Kolter. *CVPR 2024.* [[pdf](https://arxiv.org/abs/2404.07177)]

**LAB: Large-Scale Alignment for ChatBots**
 Shivchander Sudalairaj, Abhishek Bhandwaldar, Aldo Pareja, Kai Xu, David D. Cox, Akash Srivastava. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.01081)]

**Entropy Law: The Story Behind Data Compression and LLM Performance**
 Mingjia Yin, Chuhan Wu, Yufei Wang, Hao Wang, Wei Guo, Yasheng Wang, Yong Liu, Ruiming Tang, Defu Lian, Enhong Chen. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2407.06645)]

**G-DIG: Towards Gradient-based DIverse and hiGh-quality Instruction Data Selection for Machine Translation**
 Xingyuan Pan, Luyang Huang, Liyan Kang, Zhicheng Liu, Yu Lu, Shanbo Cheng. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.821.pdf)]

**Cold-Start Data Selection for Few-shot Language Model Fine-tuning: A Prompt-Based Uncertainty Propagation Approach**
 Yue Yu, Rongzhi Zhang, Ran Xu, Jieyu Zhang, Jiaming Shen, Chao Zhang. *ACL 2023.* [[pdf](https://doi.org/10.48550/arXiv.2209.06995)]

**MoDS: Model-oriented Data Selection for Instruction Tuning**
 Qianlong Du, Chengqing Zong, Jiajun Zhang. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2311.15653)]

**SemDeDup: Data-efficient learning at web-scale through semantic deduplication**
 Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, Ari S. Morcos. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2303.09540)]

**NLU on Data Diets: Dynamic Data Subset Selection for NLP Classification Tasks**
 Jean-michel Attendu, Jean-philippe Corbeil. *SustaiNLP 2023.* [[pdf](https://aclanthology.org/2023.sustainlp-1.9/)]

**When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale**
 Max Marion, Ahmet Üstün, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, Sara Hooker. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2309.04564)]

**Emergent and Predictable Memorization in Large Language Models**
 Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, Hailey Schoelkopf, Quentin Anthony, Shivanshu Purohit, Edward Raff. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2304.11158)]

**Data Pruning via Moving-one-Sample-out**
 Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, Xiaojuan Qi. *NeurIPS 2023.* [[pdf](https://arxiv.org/abs/2310.14664)]

**Instruction Mining: Instruction Data Selection for Tuning Large Language Models**
 Yihan Cao, Yanbin Kang, Chi Wang, Lichao Sun. *arxiv 2023.* [[pdf](https://arxiv.org/abs/2307.06290)]

**Rethinking the Instruction Quality: LIFT is What You Need**
 Yang Xu, Yongqiang Yao, Yufan Huang, Mengnan Qi, Maoquan Wang, Bin Gu, Neel Sundaresan. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2312.11508)]

**Biases in Large Language Models: Origins, Inventory, and Discussion**
 Roberto Navigli, Simone Conia, Björn Ross. *ACM JDIQ, 2023.* [[pdf](https://doi.org/10.1145/3597307)]

**Baichuan 2: Open Large-scale Language Models**
 Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, et al. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2309.10305)]

**Analyzing Leakage of Personally Identifiable Information in Language Models**
 Nils Lukas, Ahmed Salem, Robert Sim, Shruti Tople, Lukas Wutschitz, Santiago Zanella-Béguelin. *IEEE S&P 2023.* [[pdf](https://arxiv.org/abs/2302.00539)]

**DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4**
 Zhengliang Liu, Yue Huang, Xiaowei Yu, Lu Zhang, Zihao Wu, Chao Cao, Haixing Dai, Lin Zhao, Yiwei Li, Peng Shu, Fang Zeng, Lichao Sun, Wei Liu, Dinggang Shen, Quanzheng Li, Tianming Liu, Dajiang Zhu, Xiang Li. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2303.11032)]

**Economic Hyperparameter Optimization With Blended Search Strategy**
 Chi Wang, Qingyun Wu, Silu Huang, Amin Saied. *ICLR 2021.* [[pdf](https://iclr.cc/virtual/2021/poster/3052)]

**FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP**
 Alan Akbik, Tanja Bergmann, Duncan Blythe, Kashif Rasul, Stefan Schweter, Roland Vollgraf. *NAACL 2019.* [[pdf](https://aclanthology.org/N19-4010/)]

**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
 Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. *NAACL 2019.* [[pdf](https://aclanthology.org/N19-1423.pdf)]

**Active Learning for Convolutional Neural Networks: A Core-Set Approach**
 Ozan Sener, Silvio Savarese. *ICLR 2018.* [[pdf](https://doi.org/10.48550/arXiv.1708.00489)]

**Annotating longitudinal clinical narratives for de-identification: The 2014 i2b2/UTHealth corpus**
 Amber Stubbs, Özlem Uzuner. *J. Biomed. Inform 2015.* [[pdf](https://www.sciencedirect.com/science/article/pii/S1532046415001823)]

### 1.4 Data Transformation for LLM

**Mix-CPT: A Domain Adaptation Framework via Decoupling Knowledge Learning and Format Alignment**
 Jinhao Jiang, Junyi Li, Wayne Xin Zhao, Yang Song, Tao Zhang, Ji-Rong Wen. *ICLR 2025.* [[pdf](https://iclr.cc/virtual/2025/poster/28784)]

**Data-Juicer: A One-Stop Data Processing System for Large Language Models**
 Daoyuan Chen, Yilun Huang, Zhijian Ma, Hesen Chen, Xuchen Pan, Ce Ge, Dawei Gao, Yuexiang Xie, Zhaoyang Liu, Jinyang Gao, Yaliang Li, Bolin Ding, Jingren Zhou. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3626246.3653385)]

**DecorateLM: Data Engineering through Corpus Rating, Tagging, and Editing with Language Models**
 Ranchi Zhao, Zhen Leng Thai, Yifan Zhang, Shengding Hu, Jie Zhou, Yunqi Ba, Jie Cai, Zhiyuan Liu, Maosong Sun. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.emnlp-main.83/)]

**MM1: Methods, Analysis and Insights from Multimodal LLM Pre-training**
 Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, et al. *ECCV 2024.* [[pdf](https://arxiv.org/abs/2403.09611)]

**ShareGPT4V: Improving Large Multi-modal Models with Better Captions**
 Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, Dahua Lin. *ECCV 2024.* [[pdf](https://arxiv.org/abs/2311.12793)]

**VeCLIP: Improving CLIP Training via Visual-Enriched Captions**
 Zhengfeng Lai, Haotian Zhang, Bowen Zhang, Wentao Wu, Haoping Bai, Aleksei Timofeev, Xianzhi Du, Zhe Gan, Jiulong Shan, Chen-Nee Chuah, Yinfei Yang, Meng Cao. *ECCV 2024.* [[pdf](https://arxiv.org/abs/2310.07699)]

**Dense X Retrieval: What Retrieval Granularity Should We Use?**
 Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, Dong Yu. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.emnlp-main.845/)]

**Scalable and Domain-General Abstractive Proposition Segmentation**
 Mohammad Javad Hosseini, Yang Gao, Tim Baumgärtner, Alex Fabrikant, Reinald Kim Amplayo. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.findings-emnlp.517/)]

**Thread: A Logic-Based Data Organization Paradigm for How-To Question Answering with Retrieval Augmented Generation**
 Kaikai An, Fangkai Yang, Liqun Li, Junting Lu, Sitao Cheng, Shuzheng Si, Lu Wang, Pu Zhao, Lele Cao, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Qi Zhang, Baobao Chang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2406.13372)]

**A Hierarchical Context Augmentation Method to Improve Retrieval-Augmented LLMs on Scientific Papers**
 Tian-Yi Che, Xian-Ling Mao, Tian Lan, Heyan Huang. *KDD 2024.* [[pdf](https://doi.org/10.1145/3637528.3671847)]

**From Supervised to Generative: A Novel Paradigm for Tabular Deep Learning with Large Language Models**
 Xumeng Wen, Han Zhang, Shun Zheng, Wei Xu, Jiang Bian. *KDD 2024.* [[pdf](https://arxiv.org/abs/2310.07338v4)]

**UniK-QA: Unified Representations of Structured and Unstructured Knowledge for Open-Domain Question Answering**
 Barlas Oguz, Xilun Chen, Vladimir Karpukhin, Stan Peshterliev, Dmytro Okhonko, Michael Schlichtkrull, Sonal Gupta, Yashar Mehdad, Scott Yih. *NAACL 2022.* [[pdf](https://aclanthology.org/2022.findings-naacl.115/)]

**Learning Transferable Visual Models From Natural Language Supervision**
 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. *ICML 2021.* [[pdf](https://proceedings.mlr.press/v139/radford21a.html)]

**Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision**
 Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, Tom Duerig. *ICML 2021.* [[pdf](https://proceedings.mlr.press/v139/jia21b.html)]



### 1.5 Data Selection for LLM

**Improving Pretraining Data Using Perplexity Correlations**
 Tristan Thrush, Christopher Potts, Tatsunori Hashimoto. *ICLR 2025.* [[pdf](https://iclr.cc/virtual/2025/poster/28733)]

**Harnessing Diversity for Important Data Selection in Pretraining Large Language Models**
 Chi Zhang, Huaping Zhong, Kuan Zhang, Chengliang Chai, Rui Wang, Xinlin Zhuang, Tianyi Bai, Jiantao Qiu, Lei Cao, Ju Fan, Ye Yuan, Guoren Wang, Conghui He. *ICLR 2025.* [[pdf](https://iclr.cc/virtual/2025/poster/29114)]

**SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models**
 Yu Yang, Siddhartha Mishra, Jeffrey Chiang, Baharan Mirzasoleiman. *NeurIPS 2024.* [[pdf](https://neurips.cc/virtual/2024/poster/95679)]

**LESS: Selecting Influential Data for Targeted Instruction Tuning**
 Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen. *ICML 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.04333)]

**Efficient Continual Pre-training for Building Domain Specific Large Language Models**
 Yong Xie, Karan Aggarwal, Aitzaz Ahmad. *ACL 2024.* [[pdf](https://aclanthology.org/2024.findings-acl.606/)]

**Enabling On-Device Large Language Model Personalization with Self-Supervised Data Selection and Synthesis**
 Ruiyang Qin, Jun Xia, Zhenge Jia, Meng Jiang, Ahmed Abbasi, Peipei Zhou, Jingtong Hu, Yiyu Shi. *DAC 2024.* [[pdf](https://doi.org/10.1145/3649329.3655665)]

**Autonomous Data Selection with Language Models for Mathematical Texts**
 Yifan Zhang, Yifan Luo, Yang Yuan, Andrew Chi-Chih Yao. *ICLR 2024.* [[pdf](https://arxiv.org/abs/2402.07625)]

**How to Train Data-Efficient LLMs**
 Noveen Sachdeva, Benjamin Coleman, Wang-Cheng Kang, Jianmo Ni, Lichan Hong, Ed H. Chi, James Caverlee, Julian McAuley, Derek Zhiyuan Cheng. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2402.09668)]

**ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning**
 Yang Wu, Huayi Zhang, Yizheng Jiao, Lin Ma, Xiaozhong Liu, Jinhong Yu, Dongyu Zhang, Dezhi Yu, Wei Xu. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2412.00631)]

**Data Acquisition for Improving Model Confidence**
 Yifan Li, Xiaohui Yu, Nick Koudas. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3654934)]

**Influential Language Data Selection via Gradient Trajectory Pursuit**
 Zhiwei Deng, Tao Li, Yang Li. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2410.16710)]

**Most Influential Subset Selection: Challenges, Promises, and Beyond**
 Yuzheng Hu, Pingbang Hu, Han Zhao, Jiaqi W. Ma. *NeurIPS 2024.* [[pdf](https://arxiv.org/abs/2409.18153)]

**CoLoR-Filter: Conditional Loss Reduction Filtering for Targeted Language Model Pre-training**
 David Brandfonbrener, Hanlin Zhang, Andreas Kirsch, Jonathan Richard Schwarz, Sham Kakade. *NeurIPS 2024.* [[pdf](https://arxiv.org/abs/2406.10670v3)]

**DSDM: model-aware dataset selection with datamodels**
 Logan Engstrom, Axel Feldmann, Aleksander Mądry. *ICML 2024.* [[pdf](https://dl.acm.org/doi/10.5555/3692070.3692568)]

**Influence Scores at Scale for Efficient Language Data Sampling**
 Nikhil Anand, Joshua Tan, Maria Minakova. *EMNLP 2023.* [[pdf](https://aclanthology.org/2023.emnlp-main.152)]

**Data Selection for Language Models via Importance Resampling**
 Sang Michael Xie, Shibani Santurkar, Tengyu Ma, Percy Liang. *NeurIPS 2023.* [[pdf](https://doi.org/10.48550/arXiv.2302.03169)]

**Active Learning Principles for In-Context Learning with Large Language Models**
 Katerina Margatina, Timo Schick, Nikolaos Aletras, Jane Dwivedi-Yu. *EMNLP 2023.* [[pdf](https://aclanthology.org/2023.findings-emnlp.334)]

**D4: Improving LLM Pretraining via Document De-Duplication and Diversification**
 Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari Morcos. *NeurIPS 2023.* [[pdf](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a8f8cbd7f7a5fb2c837e578c75e5b615-Abstract-Datasets_and_Benchmarks.html)]

**Which Examples to Annotate for In-Context Learning? Towards Effective and Efficient Selection**
 Costas Mavromatis, Balasubramaniam Srinivasan, Zhengyuan Shen, Jiani Zhang, Huzefa Rangwala, Christos Faloutsos, George Karypis. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2310.20046)]

**DavIR: Data Selection via Implicit Reward for Large Language Models**
 Haotian Zhou, Tingkai Liu, Qianli Ma, Yufeng Zhang, Jianbo Yuan, Pengfei Liu, Yang You, Hongxia Yang. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2310.13008v2)]

**Datamodels: Understanding Predictions with Data and Data with Predictions**
 Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry. *ICML 2022.* [[pdf](https://proceedings.mlr.press/v162/ilyas22a.html)]

**RETRIEVE: coreset selection forsemi-supervised learning**
 Krishnateja Killamsetty, Xujiang Zhao, Feng Chen, Rishabh Iyer. *NeurIPS 2021.* [[pdf](https://dl.acm.org/doi/10.5555/3540261.3541371)]

**Bag of Tricks for Efficient Text Classification**
 Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov. *EACL 2017.* [[pdf](https://aclanthology.org/E17-2068.pdf)]



### 1.6 Data Mixing for LLM

**Task-level Distributionally Robust Optimization for Large Language Model-based Dense Retrieval**
 Guangyuan Ma, Yongliang Ma, Xing Wu, Zhenpeng Su, Ming Zhou, Songlin Hu. *AAAI 2025.* [[pdf](https://arxiv.org/abs/2408.10613)]

**RegMix: Data Mixture as Regression for Language Model Pre-training**
 Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, Min Lin. *ICLR 2025.* [[pdf](https://iclr.cc/virtual/2025/poster/30960)]

**Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training**
 Zheheng Luo, Xin Zhang, Xiao Liu, Haoling Li, Yeyun Gong, Chen Qi, Peng Cheng. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2411.14318)]

**BiMix: Bivariate Data Mixing Law for Language Model Pretraining**
 Ce Ge, Zhijian Ma, Daoyuan Chen, Yaliang Li, Bolin Ding. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2405.14908)]

**CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training of Language Models**
 Jiawei Gu, Zacc Yang, Chuanghao Ding, Rui Zhao, Fei Tan. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.emnlp-main.903)]

**Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance**
 Jiasheng Ye, Peiju Liu, Tianxiang Sun, Yunhua Zhou, Jun Zhan, Xipeng Qiu. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2403.16952)]

**Maximize Your Data's Potential: Enhancing LLM Accuracy with Two-Phase Pretraining**
 Steven Feng, Shrimai Prabhumoye, Kezhi Kong, Dan Su, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2412.15285)]

**Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models**
 Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, Hao Wang. *COLING 2024.* [[pdf](https://arxiv.org/abs/2403.03432v1)]

**Scalable Data Ablation Approximations for Language Models through Modular Training and Merging**
 Clara Na, Ian Magnusson, Ananya Harsh Jha, Tom Sherborne, Emma Strubell, Jesse Dodge, Pradeep Dasigi. *EMNLP 2024.* [[pdf](https://arxiv.org/abs/2410.15661v1)]

**Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models**
 Minghao Wu, Thuy-Trang Vu, Lizhen Qu, Reza Haf. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.emnlp-main.787)]

**ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting**
 Rui Pan, Jipeng Zhang, Xingyuan Pan, Renjie Pi, Xiaoyu Wang, Tong Zhang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2406.19976)]

**D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models**
 Haoran Que, Jiaheng Liu, Ge Zhang, Chenchen Zhang, Xingwei Qu, Yinghao Ma, Feiyu Duan, Zhiqi Bai, Jiakai Wang, Yuanxing Zhang, Xu Tan, Jie Fu, Wenbo Su, Jiamang Wang, Lin Qu, Bo Zheng. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2406.01375)]

**Data Proportion Detection for Optimized Data Management for Large Language Models**
 Hao Liang, Keshi Zhao, Yajie Yang, Bin Cui, Guosheng Dong, Zenan Zhou, Wentao Zhang. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2409.17527)]

**DoGE: Domain Reweighting with Generalization Estimation**
 Simin Fan, Matteo Pagliardini, Martin Jaggi. *ICML 2024.* [[pdf](https://doi.org/10.48550/arXiv.2310.15393)]

**DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining**
 Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu. *NeurIPS 2023.* [[pdf](https://arxiv.org/abs/2305.10429)]

**SlimPajama-DC: Understanding Data Combinations for LLM Training**
 Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Zhengzhong Liu, Hongyi Wang, Bowen Tan, Joel Hestness, Natalia Vassilieva, Daria Soboleva, Eric Xing. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2309.10818)]

**Efficient Online Data Mixing For Language Model Pre-Training**
 Alon Albalak, Liang-Ming Pan, Colin Raffel, William Yang Wang. *NeurIPS 2023.* [[pdf](https://nips.cc/virtual/2023/81179)]

**Qwen Technical Report**
 Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, et al. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2309.16609v1)]

**LightGBM: a highly efficient gradient boosting decision tree**
 Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. *NeurIPS 2017.* [[pdf](https://dl.acm.org/doi/10.5555/3294996.3295074)]

**An overview of bilevel optimization**
 Benoît Colson, Patrice Marcotte, Gilles Savard. *AOR 2007.* [[pdf](https://link.springer.com/article/10.1007/s10479-007-0176-2)]



### 1.7 Data Synthesis and Augmentation for LLM

**Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling**
 Zhenyu Hou, Xin Lv, Rui Lu, Jiajie Zhang, Yujiang Li, Zijun Yao, Juanzi Li, Jie Tang, Yuxiao Dong. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.11651)]

**LLMs Can Easily Learn to Reason from Demonstrations: Structure, Not Content, Is What Matters!**
 Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Eric Tang, Sumanth Hegde, Kourosh Hakhamaneshi, Shishir G. Patil, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.07374)]

**Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search**
 Maohao Shen, Guangtao Zeng, Zhenting Qi, Zhang-Wei Hong, Zhenfang Chen, Wei Lu, Gregory Wornell, Subhro Das, David Cox, Chuang Gan. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.02508)]

**LIMO: Less is More for Reasoning**
 Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, Pengfei Liu. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.03387)]

**LLM See, LLM Do: Leveraging Active Inheritance to Target Non-Differentiable Objectives**
 Luísa Shimabucoro, Sebastian Ruder, Julia Kreutzer, Marzieh Fadaee, Sara Hooker. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.emnlp-main.521)]

**Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models**
 Haoran Li, Qingxiu Dong, Zhengyang Tang, et al. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.13064)]

**Mitigating the Privacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data**
 Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren, Tianqi Zheng, Hanqing Lu, Han Xu, Hui Liu, Yue Xing, Jiliang Tang. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2406.14773)]

**Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning**
 Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, Weizhu Chen. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.02333)]

**Augmenting Math Word Problems via Iterative Question Composing**
 Haoxiong Liu, Yifan Zhang, Yifan Luo, Andrew Chi-Chih Yao. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.09003)]

**MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data**
 Yinya Huang, Xiaohan Lin, Zhengying Liu, Qingxing Cao, Huajian Xin, Haiming Wang, Zhenguo Li, Linqi Song, Xiaodan Liang. *ICLR 2024.* [[pdf](https://arxiv.org/abs/2402.08957v3)]

**AgentInstruct: Toward Generative Teaching with Agentic Flows**
 Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andres Codas, Yadong Lu, Wei-ge Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, Ahmed Awadallah. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2407.03502)]

**JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models**
 Kun Zhou, Beichen Zhang, Jiapeng Wang, Zhipeng Chen, Wayne Xin Zhao, Jing Sha, Zhichao Sheng, Shijin Wang, Ji-Rong Wen. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2405.14365)]

**Differentially Private Synthetic Data via Foundation Model APIs 2: Text**
 Chulin Xie, Zinan Lin, Arturs Backurs, Sivakanth Gopi, Da Yu, Huseyin A Inan, Harsha Nori, Haotian Jiang, Huishuai Zhang, Yin Tat Lee, Bo Li, Sergey Yekhanin. *ICML 2024.* [[pdf](https://arxiv.org/abs/2403.01749v2)]

**WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions**
 Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, Daxin Jiang. *ICLR 2024.* [[pdf](https://iclr.cc/virtual/2024/poster/19164)]

**Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations**
 Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, Zhifang Sui. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.510)]

**Self-Instruct: Aligning Language Models with Self-Generated Instructions**
 Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. *ACL 2023.* [[pdf](https://aclanthology.org/2023.acl-long.754)]

**Privacy-Preserving In-Context Learning with Differentially Private Few-Shot Generation**
 Xinyu Tang, Richard Shin, Huseyin A. Inan, Andre Manoel, Fatemehsadat Mireshghallah, Zinan Lin, Sivakanth Gopi, Janardhan Kulkarni, Robert Sim. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2309.11765)]

**Large Language Models are Human-Level Prompt Engineers**
 Yongchao Zhou, Andrei Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, Jimmy Ba. *ICLR 2023.* [[pdf](https://iclr.cc/virtual/2023/poster/10850)]

**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**
 Lianmin Zheng, Wei - Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. *NeurIPS 2023.* [[pdf](https://arxiv.org/abs/2306.05685v4)]

**Let's Verify Step by Step**
 Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2305.20050)]

**Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**
 Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, et al. *arXiv 2022.* [[pdf](https://doi.org/10.48550/arXiv.2204.05862)]


### 1.8 Data Processing Pipelines for LLM

**Data-Juicer: A One-Stop Data Processing System for Large Language Models**
 Daoyuan Chen, Yilun Huang, Zhijian Ma, Hesen Chen, Xuchen Pan, Ce Ge, Dawei Gao, Yuexiang Xie, Zhaoyang Liu, Jinyang Gao, Yaliang Li, Bolin Ding, Jingren Zhou. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3626246.3653385)]

**Dataverse: Open-Source ETL (Extract, Transform, Load) Pipeline for Large Language Models**
 Hyunbyung Park, Sukyung Lee, Gyoungjin Gim, Yungi Kim, Dahyun Kim, Chanjun Park. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2403.19340v1)]

**An Integrated Data Processing Framework for Pretraining Foundation Models**
 Yiding Sun, Feng Wang, Yutao Zhu, Wayne Xin Zhao, Jiaxin Mao. *SIGIR 2024.* [[pdf](https://doi.org/10.1145/3626772.3657671)]

**SAGE: A Framework of Precise Retrieval for RAG**
*Jintao Zhang, Guoliang Li, Jinyang Su. ICDE 2025.* [[pdf](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/ICDE25-SAGE.pdf)]

**Oasis: data curation and assessment system for pretraining of large language models**
 Tong Zhou, Yubo Chen, Pengfei Cao, Kang Liu, Shengping Liu, Jun Zhao. *IJCAI 2024.* [[pdf](https://doi.org/10.24963/ijcai.2024/1048)]

**Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development**
 Daoyuan Chen, Haibin Wang, Yilun Huang, Ce Ge, Yaliang Li, Bolin Ding, Jingren Zhou. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2407.11784v2)]

**LP Data Pipeline: Lightweight, Purpose-driven Data Pipeline for Large Language Models**
 Yungi Kim, Hyunsoo Ha, Seonghoon Yang, Sukyung Lee, Jihoo Kim, Chanjun Park. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2411.11289v1)]

**The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale**
 Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf. *NeurIPS 2024.* [[pdf](https://papers.neurips.cc/paper_files/paper/2024/file/370df50ccfdf8bde18f8f9c2d9151bda-Paper-Datasets_and_Benchmarks_Track.pdf)]

**DataComp-LM: In search of the next generation of training sets for language models**
 Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, et al. *NeurIPS 2024.* [[pdf](https://arxiv.org/abs/2406.11794v3)]

**Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset**
 Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2412.02595v1)]

**LLaMA: Open and Efficient Foundation Language Models**
 Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie - Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2302.13971v1)]

**The RefinedWeb dataset for falcon LLM: outperforming curated corpora with web data only**
 Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, Julien Launay. *NeurIPS 2023.* [[pdf](https://arxiv.org/abs/2306.01116v1)]

**Baichuan 2: Open Large-scale Language Models**
 Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, et al. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2309.10305)]

**Scaling Language Models: Methods, Analysis & Insights from Training Gopher**
 Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, et al. *arXiv 2021.* [[pdf](https://arxiv.org/abs/2112.11446v2)]

**Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction**
 Adrien Barbaresi. *ACL 2021.* [[pdf](https://aclanthology.org/2021.acl-demo.15.pdf)]

**Exploring the limits of transfer learning with a unified text-to-text transformer**
 Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. *JMLR 2020.* [[pdf](https://arxiv.org/abs/1910.10683v4)]

**CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data**
 Guillaume Wenzek, Marie - Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, Edouard Grave. *LREC 2020.* [[pdf](https://aclanthology.org/2020.lrec-1.494/)]

**Removing Boilerplate and Duplicate Content from Web Corpora**
 Jan Pomikálek. Doctoral dissertation(Masaryk University, Brno, Czech Republic) 2011. [[pdf](https://docslib.org/doc/706394/removing-boilerplate-and-duplicate-content-from-web-corpora)]



### 1.9 Utilities

#### 1.9.1 Data Provenance for LLM

**Provable Robust Watermarking for AI-Generated Text**
 Xuandong Zhao, Prabhanjan Ananth, Lei Li, Yu-Xiang Wang. *ICLR 2024.* [[pdf](https://arxiv.org/pdf/2306.17439v2.pdf)]

**An Unforgeable Publicly Verifiable Watermark for Large Language Models**
 Aiwei Liu, Leyi Pan, Xuming Hu, Shu'ang Li, Lijie Wen, Irwin King, Philip S. Yu. *ICLR 2024.* [[pdf](https://arxiv.org/pdf/2307.16230v7.pdf)]

**Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**
 Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren. *NeurIPS 2024.* [[pdf](https://arxiv.org/pdf/2406.01946v3.pdf)]

**Undetectable Watermarks for Language Models**
 Miranda Christ, Sam Gunn, Or Zamir. *COLT 2024.* [[pdf](https://proceedings.mlr.press/v247/christ24a.html)]

**A Watermark for Large Language Models**
 John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein. *ICML 2023.* [[pdf](https://arxiv.org/abs/2301.10226v4)]

**Publicly-Detectable Watermarking for Language Models**
 Jaiden Fairoze, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Mingyuan Wang. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2310.18491)]


#### 1.9.2 Data Visualization for LLM

**Data-Juicer: A One-Stop Data Processing System for Large Language Models**
Daoyuan Chen, Yilun Huang, Zhijian Ma, Hesen Chen, Xuchen Pan, Ce Ge, Dawei Gao, Yuexiang Xie, Zhaoyang Liu, Jinyang Gao, Yaliang Li, Bolin Ding, Jingren Zhou. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3626246.3653385)]


#### 1.9.3 Constructing Dense LLMs

**MiniMax-01: Scaling Foundation Models with Lightning Attention**
 MiniMax, Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, et al. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.08313)]

**Process Reinforcement through Implicit Rewards**
 Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, Ning Ding. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.01456)]

**Densing Law of LLMs**
 Chaojun Xiao, Jie Cai, Weilin Zhao, Guoyang Zeng, Biyuan Lin, Jie Zhou, Zhi Zheng, Xu Han, Zhiyuan Liu, Maosong Sun. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2412.04315)]

**InternLM2 Technical Report**
 Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, et al. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.17297)]

**DeepSeek-V3 Technical Report**
 DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, et al. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2412.19437)]

**MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies**
 Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, et al. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2404.06395)]

**Mixtral of Experts**
 Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, et al. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.04088)]

**Linearizing Large Language Models**
 Jean Mercat, Igor Vasiljevic, Sedrick Keh, Kushal Arora, Achal Dave, Adrien Gaidon, Thomas Kollar. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.06640)]

**Mistral 7B**
 Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2310.06825)]

**GPT-4 Technical Report**
 OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, et al. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2303.08774)]

**LLaMA: Open and Efficient Foundation Language Models**
 Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, et al. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2302.13971)]

**Training compute-optimal large language models**
 Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. *NeurIPS 2022.* [[pdf](https://dl.acm.org/doi/10.5555/3600270.3602446)]

**Scaling Laws for Neural Language Models**
 Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei. *arXiv 2020.* [[pdf](https://arxiv.org/abs/2001.08361)]


## 2. Data Storage for LLM

### 2.1 Data Storage for Training

*2.1.1 Training Data Storage*

**cedar: Optimized and Unified Machine Learning Input Data Pipelines**
 Mark Zhao, Emanuel Adamiak, Christos Kozyrakis. *VLDB 2025.* [[pdf](https://doi.org/10.48550/arXiv.2401.08895)]

**CC - GPX: Extracting High-Quality Annotated Geospatial Data from Common Crawl**
 Ilya Ilyankou, Meihui Wang, Stefano Cavazzi, James Haworth. *SIGSPATIAL 2024.* [[pdf](https://doi.org/10.1145/3678717.3691215)]

**Pecan: cost-efficient ML data preprocessing with automatic transformation ordering and hybrid placement**
 Dan Graur, Oto Mraz, Muyu Li, Sepehr Pourghannad, Chandramohan A. Thekkath, Ana Klimovic. *USENIX ATC 2024.* [[pdf](https://dl.acm.org/doi/10.5555/3691992.3692032)]

**The Image Calculator: 10x Faster Image-AI Inference by Replacing JPEG with Self-designing Storage Format**
 Utku Sirin, Stratos Idreos. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3639307)]

**tf.data service: A Case for Disaggregating ML Input Data Processing**
 Andrew Audibert, Yang Chen, Dan Graur, Ana Klimovic, Jiří Šimša, Chandramohan A. Thekkath. *SoCC 2023.* [[pdf](https://doi.org/10.1145/3620678.3624666)]

**SUFS: A Generic Storage Usage Forecasting Service Through Adaptive Ensemble Learning**
 Luming Sun, Shijin Gong, Tieying Zhang, Fuxin Jiang, Zhibing Zhao, Jianjun Chen. *ICDE 2023.* [[pdf](https://ieeexplore.ieee.org/document/10184683)]

**SiloD: A Co-design of Caching and Scheduling for Deep Learning Clusters**
 Hanyu Zhao, Zhenhua Han, Zhi Yang, Quanlu Zhang, Mingxia Li, Fan Yang, Qianxi Zhang, Binyang Li, Yuqing Yang, Lili Qiu, Lintao Zhang, Lidong Zhou. *EuroSys 2023.* [[pdf](https://doi.org/10.1145/3552326.3567499)]

**Tectonic-Shift: A Composite Storage Fabric for Large-Scale ML Training**
 Mark Zhao, Satadru Pan, Niket Agarwal, Zhaoduo Wen, David Xu, Anand Natarajan, Pavan Kumar, Shiva Shankar P, Ritesh Tijoriwala, Karan Asher, Hao Wu, Aarti Basant, Daniel Ford, Delia David, Nezih Yigitbasi, Pratap Singh, Carole-Jean Wu, Christos Kozyrakis. *USENIX ATC 2023.* [[pdf](https://www.usenix.org/conference/atc23/presentation/zhao)]

**Fluid: Dataset Abstraction and Elastic Acceleration for Cloud-native Deep Learning Training Jobs**
 Rong Gu, Kai Zhang, Zhihao Xu, Yang Che, Bin Fan, Haojun Hou. *ICDE 2022.* [[pdf](https://doi.org/10.1109/ICDE53745.2022.00209)]

**Cachew: Machine Learning Input Data Processing as a Service**
 Dan Graur, Damien Aymon, Dan Kluser, Tanguy Albrici, Chandramohan A. Thekkath, Ana Klimovic. *USENIX ATC 2022.* [[pdf](https://www.usenix.org/conference/atc22/presentation/graur)]

**An Overview of Data Warehouse and Data Lake in Modern Enterprise Data Management**
 Athira Nambiar, Divyansh Mundra. *BDCC 2022.* [[pdf](https://doi.org/10.3390/bdcc6040132)]

**Quiver: An Informed Storage Cache for Deep Learning**
 Abhishek Kumar, Muthian Sivathanu. *USENIX FAST 2020.* [[pdf](https://www.usenix.org/conference/fast20/presentation/kumar)]

**I/O Characterization and Performance Evaluation of BeeGFS for Deep Learning**
 Fahim Chowdhury, Yue Zhu, Todd Heer, Saul Paredes, Adam Moody, Robin Goldstone, Kathryn Mohror, Weikuan Yu. *ICPP 2019.* [[pdf](https://doi.org/10.1145/3337821.3337902)]

**I/O Bottleneck Investigation in Deep Learning Systems**
 S. Pumma, Min Si, Wu-chun Feng, P. Balaji. *ICPP 2018.* [[pdf](https://www.semanticscholar.org/paper/I-O-Bottleneck-Investigation-in-Deep-Learning-Pumma-Si/e1486bf2783f5da4cf9a3785f44c7efdb0793c33)]

**High Performance I/O**
 Adrian Jackson, Fiona Reid, Joachim Hein, Alejandro Soba, Xavier Saez. *Euromicro PDP 2011.* [[pdf](https://ieeexplore.ieee.org/abstract/document/5739034)]

*2.1.2 Model Data Storage*

**An Empirical Study of Safetensors' Usage Trends and Developers' Perceptions**
 Beatrice Casey, Kaia Damian, Andrew Cotaj, Joanna C. S. Santos. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.02170)]

**MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs**
 Ziheng Jiang, Haibin Lin, Yinmin Zhong, Qi Huang, et al. *USENIX NSDI 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.15627)]

**ProTrain: Efficient LLM Training via Memory-Aware Techniques**
 Hanmei Yang, Jin Zhou, Yao Fu, Xiaoqun Wang, Ramine Roane, Hui Guan, Tongping Liu. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2406.08334)]

**ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development**
 Borui Wan, Mingji Han, Yiyao Sheng, Yanghua Peng, Haibin Lin, Mofan Zhang, Zhichao Lai, Menghan Yu, Junda Zhang, Zuquan Song, Xin Liu, Chuan Wu. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2407.20143)]

**GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints**
 Zhuang Wang, Zhen Jia, Shuai Zheng, Zhen Zhang, Xinwei Fu, T. S. Eugene Ng, Yida Wang. *SOSP 2023.* [[pdf](https://doi.org/10.1145/3600006.3613145)]

**CheckFreq: Frequent, Fine-Grained DNN Checkpointing**
 Jayashree Mohan, Amar Phanishayee, Vijay Chidambaram. *USENIX FAST 2021.* [[pdf](https://www.usenix.org/conference/fast21/presentation/mohan)]

**ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning**
 Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He. *SC 2021.* [[pdf](https://doi.org/10.1145/3458817.3476205)]

**ZeRO-Offload: Democratizing Billion-Scale Model Training**
 Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, Yuxiong He. *USENIX ATC 2021.* [[pdf](https://www.researchgate.net/profile/Jie-Ren-14/publication/348589607_ZeRO-Offload_Democratizing_Billion-Scale_Model_Training/links/602c2c2c92851c9287908616/ZeRO-Offload-Democratizing-Billion-Scale-Model-Training.pdf)]

**ZeRO: memory optimizations toward training trillion parameter models**
 Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. *SC 2020.* [[pdf](https://dl.acm.org/doi/10.5555/3433701.3433727)]

**vDNN: virtualized deep neural networks for scalable, memory-efficient neural network design**
 Minsoo Rhu, Natalia Gimelshein, Jason Clemons, Arslan Zulfiqar, Stephen W. Keckler. *MICRO-49 2016.* [[pdf](https://dl.acm.org/doi/10.5555/3195638.3195660)]

### 2.2 Data Storage for Inference

**Fast State Restoration in LLM Serving with HCache**
 Shiwei Gao, Youmin Chen, Jiwu Shu. *EuroSys 2025.* [[pdf](https://arxiv.org/abs/2410.05004)]

**Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention**
 Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu, Pengfei Zuo. *USENIX ATC 2024.* [[pdf](https://arxiv.org/abs/2403.19708)]

**MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool**
 Cunchen Hu, Heyang Huang, Junhao Hu, Jiang Xu, Xusheng Chen, Tao Xie, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, Yizhou Shan. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2406.17565)]

**Efficient LLM Inference with I/O-Aware Partial KV Cache Recomputation**
 Chaoyi Jiang, Lei Gao, Hossein Entezari Zarch, Murali Annavaram. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2411.17089)]

**QUEST: query-aware sparsity for efficient long-context LLM inference**
 Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han. *ICML 2024.* [[pdf](https://dl.acm.org/doi/pdf/10.5555/3692070.3694025)]

**ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition**
 Lu Ye, Ze Tao, Yong Huang, Yang Li. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.623.pdf)]

**BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching**
 Zhen Zheng, Xin Ji, Taosong Fang, Fanghao Zhou, Chuanjie Liu, Gang Peng. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2412.03594)]

**Efficient Memory Management for Large Language Model Serving with PagedAttention**
 Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, Ion Stoica. *SOSP 2023.* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165)]

**VTensor: Using Virtual Tensors to Build a Layout-oblivious AI Programming Framework**
 Feng Yu, Jiacheng Zhao, Huimin Cui, Xiaobing Feng, Jingling Xue. *PACT 2020.* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3410463.3414664)]

### 2.3 Data Storage for RAG

**Caching of Retrieval Augmented Generation Model Trained with Dynamic Data and Dynamically Updating of Cache**
 Mustafa Kadioglu, Paul C. Stojanovski, Sunil Karthik Kota. *Technical Disclosure Commons 2024.* [[pdf](https://www.tdcommons.org/dpubs_series/7154/)]

**The Faiss Library**
 Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, Hervé Jégou. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.08281)]

**RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**
 Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, Xin Jin. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2404.12457)]

**CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**
 Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, Junchen Jiang. *SIGCOMM 2024.* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3651890.3672274)]

**TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text**
 Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, Yaohua Tang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2410.07590)]

**GleanVec: Accelerating Vector Search with Minimalist Nonlinear Dimensionality Reduction**
 Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, Ted Willke. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2410.22347)]

**CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**
 Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, Junchen Jiang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2405.16444)]

**LeanVec: Searching Vectors Faster by Making Them Fit**
 Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, Mark Hildebrand, Ted Willke. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2312.16335)]

**Graph Databases Assessment: JanusGraph, Neo4j, and TigerGraph**
 Jéssica Monteiro, Filipe Sá, Jorge Bernardino. *Perspectives and Trends in Education and Technology 2023.* [[pdf](https://doi.org/10.1007/978-981-19-6585-2_58)]

**Similarity Search in the Blink of an Eye with Compressed Indices**
 Cecilia Aguerrebere, Ishwar Singh Bhati, Mark Hildebrand, Mariano Tepper, Theodore Willke. *VLDB 2023.* [[pdf](https://doi.org/10.14778/3611479.3611537)]

**Empirical Evaluation of a Cloud-Based Graph Database: the Case of Neptune**
 Ghislain Auguste Atemezing. *KGSWC 2021.* [[pdf](https://doi.org/10.1007/978-3-030-91305-2_3)]

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
 Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela. *NeurIPS 2020.* [[pdf](https://doi.org/10.48550/arXiv.2005.11401)]

## 3. Data Serving for LLM

### 3.1 Data Serving for Training

**Structured Packing in LLM Training Improves Long Context Utilization**
 Konrad Staniszewski, Szymon Tworkowski, Sebastian Jaszczur, Yu Zhao, Henryk Michalewski, Łukasz Kuciński, Piotr Miłoś. *AAAI 2025.* [[pdf](https://doi.org/10.48550/arXiv.2312.17296)]

**Fewer Truncations Improve Language Modeling**
 Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, Stefano Soatto. *ICML 2024.* [[pdf](https://doi.org/10.48550/arXiv.2404.10830)]

**How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition**
 Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, Jingren Zhou. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.12.pdf)]

**Bucket Pre-training is All You Need**
 Hongtao Liu, Qiyao Peng, Qing Yang, Kai Liu, Hongyan Xu. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2407.07495)]

**Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum**
 Hadi Pouransari, Chun-Liang Li, Jen-Hao Rick Chang, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Oncel Tuzel. *NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.13226)]

**In-context Pretraining: Language Modeling Beyond Document Boundaries**
 Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Gergely Szilvasy, Rich James, Xi Victoria Lin, Noah A. Smith, Luke Zettlemoyer, Scott Yih, Mike Lewis. *ICLR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2310.10638)]

**Automatic Pruning of Fine-tuning Datasets for Transformer-based Language Models**
 Mohammadreza Tayaranian, Seyyed Hasan Mozafari, Brett H. Meyer, James J. Clark, Warren J. Gross. *CoLLAs 2024.* [[pdf](https://doi.org/10.48550/arXiv.2407.08887)]

**Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning**
 Jisu Kim, Juhwan Lee. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.07490)]

**NLU on Data Diets: Dynamic Data Subset Selection for NLP Classification Tasks**
 Jean-michel Attendu, Jean-philippe Corbeil. *SustaiNLP 2023.* [[pdf](https://aclanthology.org/2023.sustainlp-1.9.pdf)]

**BERT on a Data Diet: Finding Important Examples by Gradient-Based Pruning**
 Mohsen Fayyaz, Ehsan Aghazadeh, Ali Modarressi, Mohammad Taher Pilehvar, Yadollah Yaghoobzadeh, Samira Ebrahimi Kahou. *NeurIPS 2022.* [[pdf](https://doi.org/10.48550/arXiv.2211.05610)]

**Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance**
 Mario Michael Krell, Matej Kosec, Sergio P. Perez, Andrew Fitzgibbon. *arXiv 2021.* [[pdf](https://doi.org/10.48550/arXiv.2107.02027)]

**Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory**
 James L. McClelland, Bruce L. McNaughton, Randall C. O’Reilly. *Psychological Review 1995.* [[pdf](https://cseweb.ucsd.edu/~gary/258/jay.pdf)]

**Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem**
 M. McCloskey, N. J. Cohen. *Psychology of Learning and Motivation 1989.* [[pdf](https://www.sciencedirect.com/science/article/abs/pii/S0079742108605368)]

### 3.2 Data Serving for Inference

**LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression**

Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.91/)]

**CoachLM: Automatic Instruction Revisions Improve the Data Quality in LLM Instruction Tuning**

*Liu, Yilun, Shimin Tao, Xiaofeng Zhao, Ming Zhu, Wenbing Ma, Junhao Zhu, Chang Su et al. ICDE 2024.* [[pdf](https://ieeexplore.ieee.org/document/10597991)]

**LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression**

Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Dongmei Zhang. *ACL 2024.* [[pdf](https://aclanthology.org/2024.findings-acl.57.pdf)]

**Adapting Language Models to Compress Contexts**

Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen. *EMNLP 2023.* [[pdf](https://aclanthology.org/2023.emnlp-main.232.pdf)]

**LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models**

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu. *EMNLP 2023.* [[pdf](https://aclanthology.org/2023.emnlp-main.825.pdf)]

**Learning to Compress Prompts with Gist Tokens**

Jesse Mu, Xiang Lisa Li, Noah Goodman. *NeurIPS 2023.* [[pdf](https://arxiv.org/abs/2304.08467)]

### 3.3 Data Serving for RAG

**ASRank: Zero-Shot Re-Ranking with Answer Scent for Document Retrieval**
 Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Adam Jatowt. *NAACL 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.15245)]

**Context Embeddings for Efficient Answer Generation in RAG**
 David Rau, Shuai Wang, Hervé Déjean, Stéphane Clinchant. *WSDM 2025.* [[pdf](https://doi.org/10.48550/arXiv.2407.09252)]

**Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation**
 Zijie Zhong, Hanwen Liu, Xiaoya Cui, Xiaofan Zhang, Zengchang Qin. *COLING 2025.* [[pdf](https://doi.org/10.48550/arXiv.2406.00456)]

**MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation**
 Tianyu Fan, Jingyuan Wang, Xubin Ren, Chao Huang. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.06713)]

**SAGE: A Framework of Precise Retrieval for RAG**
 *Jintao Zhang, Guoliang Li, Jinyang Su. ICDE 2025.* [[pdf](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/ICDE25-SAGE.pdf)]

**xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token**
 Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, Dongyan Zhao. *NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.13792)]

**From Local to Global: A Graph RAG Approach to Query-Focused Summarization**
 Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2404.16130)]

**ARAGOG: Advanced RAG Output Grading**
 Matouš Eibich, Shivay Nagpal, Alexander Fred-Ojala. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2404.01037)]

**M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**
 Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu. *ACL 2024.* [[pdf](https://aclanthology.org/2024.findings-acl.137.pdf)]

**Familiarity-Aware Evidence Compression for Retrieval-Augmented Generation**
 Dongwon Jung, Qin Liu, Tenghao Huang, Ben Zhou, Muhao Chen. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2409.12468)]

**Grounding Language Model with Chunking-Free In-Context Retrieval**
 Hongjin Qian, Zheng Liu, Kelong Mao, Yujia Zhou, Zhicheng Dou. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.71/)]

**Compressing Long Context for Enhancing RAG with AMR-based Concept Distillation**
 Kaize Shi, Xueyao Sun, Qing Li, Guandong Xu. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.03085)]

**RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation**
 Fangyuan Xu, Weijia Shi, Eunsol Choi. *ICLR 2024.* [[pdf](https://iclr.cc/virtual/2024/poster/17885)]

**Relational Database Augmented Large Language Model**

*Zongyue Qin, Chen Luo, Zhengyang Wang, Haoming Jiang, Yizhou Sun. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2407.15071)]

**LightRAG: Simple and Fast Retrieval-Augmented Generation**
 Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2410.05779)]

**Towards General Text Embeddings with Multi-stage Contrastive Learning**
 Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2308.03281)]

**RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models**
 Ronak Pradeep, Sahel Sharifymoghaddam, Jimmy Lin. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2309.15088)]

**Learning Transferable Visual Models From Natural Language Supervision**
 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. *arXiv 2021.* [[pdf](https://arxiv.org/abs/2103.00020)]



## 4. LLM for Data Processing

### 4.1 Data Cleaning

**GIDCL: A Graph-Enhanced Interpretable Data Cleaning Framework with Large Language Models**
 Mengyi Yan, Yaoshu Wang, Yue Wang, Xiaoye Miao, Jianxin Li. *SIGMOD 2025.* [[pdf](https://dl.acm.org/doi/10.1145/3698811)]

**Mind the Data Gap: Bridging LLMs to Enterprise Data Integration**
 Moe Kayali, Fabian Wenz, Nesime Tatbul, Çağatay Demiralp. *CIDR 2025.* [[pdf](https://arxiv.org/pdf/2412.20331)]

**AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark**
 Lan Li, Liri Fang, Vetle I. Torvik. *arXiv 2024.* [[pdf](https://arxiv.org/pdf/2412.06724)]

**Jellyfish: A Large Language Model for Data Preprocessing**
 Haochen Zhang, Yuyang Dong, Chuan Xiao, Masafumi Oyamada. *arXiv 2024.* [[pdf](https://arxiv.org/pdf/2312.01678)]

**CleanAgent: Automating Data Standardization with LLM-based Agents**
 Danrui Qi, Jiannan Wang. *arXiv 2024.* [[pdf](https://arxiv.org/pdf/2403.08291)]

**LLMClean: Context-Aware Tabular Data Cleaning via LLM-Generated OFDs**
 Fabian Biester, Mohamed Abdelaal, Daniel Del Gaudio. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2404.18681)]

**LLMs with User-defined Prompts as Generic Data Operators for Reliable Data Processing**
 Luyi Ma, Nikhil Thakurdesai, Jiao Chen, Jianpeng Xu, Evren Körpeoglu, Sushant Kumar, Kannan Achan. *IEEE Big Data 2023.* [[pdf](https://arxiv.org/pdf/2312.16351)]

**SEED: Domain-Specific Data Curation With Large Language Models**
 Zui Chen, Lei Cao, Sam Madden, Tim Kraska, Zeyuan Shang, Ju Fan, Nan Tang, Zihui Gu, Chunwei Liu, Michael Cafarella. *arXiv 2023.* [[pdf](https://arxiv.org/pdf/2310.00749)]

**Large Language Models as Data Preprocessors**
 Haochen Zhang, Yuyang Dong, Chuan Xiao, Masafumi Oyamada. *arXiv 2023.* [[pdf](https://arxiv.org/pdf/2308.16361)]

**Data Cleaning Using Large Language Models**
 Shuo Zhang, Zezhou Huang, Eugene Wu. *arXiv 2024.* [[pdf](https://arxiv.org/pdf/2410.15547)]


### 4.2 Entity Matching

**Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching**
 Tianshu Wang, Hongyu Lin, Xiaoyang Chen, Xianpei Han, Hao Wang, Zhenyu Zeng, Le Sun. *COLING 2025.* [[pdf](https://arxiv.org/pdf/2405.16884)]

**Cost-Effective In-Context Learning for Entity Resolution: A Design Space Exploration**
 Meihao Fan, Xiaoyue Han, Ju Fan, Chengliang Chai, Nan Tang, Guoliang Li, Xiaoyong Du. *ICDE 2024.* [[pdf](https://arxiv.org/pdf/2312.03987)]

**In Situ Neural Relational Schema Matcher**
 Xingyu Du, Gongsheng Yuan, Sai Wu, Gang Chen, and Peng Lu. *ICDE 2024.* [[pdf](https://ieeexplore.ieee.org/abstract/document/10597805)]

**KcMF: A Knowledge-compliant Framework for Schema and Entity Matching with Fine-tuning-free LLMs**
 Yongqin Xu, Huan Li, Ke Chen, Lidan Shou. *arXiv 2024.* [[pdf](https://arxiv.org/pdf/2410.12480)]

**Unicorn: A Unified Multi-tasking Model for Supporting Matching Tasks in Data Integration**
 Jianhong Tu, Ju Fan, Nan Tang, Peng Wang, Guoliang Li, Xiaoyong Du. *SIGMOD 2023.* [[pdf](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/Unicorn_PACMMOD.pdf)]

**Entity matching using large language models**
 Ralph Peeters, Christian Bizer. *arXiv 2023.* [[pdf](https://arxiv.org/pdf/2310.11244)]

**Deep Entity Matching with Pre-Trained Language Models**
 Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, Wang-Chiew Tan. *VLDB 2021.* [[pdf](https://www.vldb.org/pvldb/vol14/p50-li.pdf)]

**Dual-Objective Fine-Tuning of BERT for Entity Matching**
 Ralph Peeters, Christian Bizer. *VLDB 2021.* [[pdf](https://madoc.bib.uni-mannheim.de/59958/1/p1913-peeters.pdf)]

### 4.3 Schema Matching

**Schema Matching with Large Language Models: an Experimental Study**

Marcel Parciak, Brecht Vandevoort, Frank Neven, Liesbet M. Peeters, Stijn Vansummeren. *VLDB 2024.* [[pdf](https://doi.org/10.48550/arXiv.2407.11852)]

**Magneto: Combining Small and Large Language Models for Schema Matching**

Yurong Liu, Eduardo Pena, Aecio Santos, Eden Wu, Juliana Freire. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2412.08194)]

**Schema Matching with Large Language Models: an Experimental Study**

Marcel Parciak, Brecht Vandevoort, Frank Neven, Liesbet M. Peeters, Stijn Vansummeren. *arxiv 2024.* [[pdf](https://arxiv.org/pdf/2407.11852)]

**Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching**

*Chuangtao Ma, Sriom Chakrabarti, Arijit Khan, Bálint Molnár*. *arxiv 2024. [[pdf](https://arxiv.org/pdf/2501.08686)]*

**KcMF: A Knowledge-compliant Framework for Schema and Entity Matching with Fine-tuning-free LLMs**

Yongqin Xu, Huan Li, Ke Chen, Lidan Shou. *arxiv 2024.* [[pdf](https://arxiv.org/pdf/2410.12480)]

### 4.4 Data Discovery

**CHORUS: Foundation Models for Unified Data Discovery and Exploration**

Moe Kayali, Anton Lykov, Ilias Fountalis, Nikolaos Vasiloglou, Dan Olteanu, Dan Suciu. *VLDB 2024.* [[pdf](https://www.vldb.org/pvldb/vol17/p2104-kayali.pdf)]

**Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes**

Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Hojel, Immanuel Trummer, Christopher Ré. *VLDB 2024.* [[pdf](https://www.vldb.org/pvldb/vol17/p92-arora.pdf)]

**DeepJoin: Joinable Table Discovery with Pre-trained Language Models**

Yuyang Dong, Chuan Xiao, Takuma Nozawa, Masafumi Enomoto, Masafumi Oyamada. *VLDB 2023.* [[pdf](https://www.vldb.org/pvldb/vol16/p2458-dong.pdf)]



## 5. LLM for Data Analysis

### 5.1 Structured Data Analysis

**Text2SQL is Not Enough: Unifying AI and Databases with TAG**

*Asim Biswal, Siddharth Jha, Carlos Guestrin, Matei Zaharia, Joseph E Gonzalez, Amog Kamsetty, Shu Liu, Liana Patel. CIDR 2025.* [[pdf](https://arxiv.org/pdf/2408.14717)]

**TableMaster: A Recipe to Advance Table Understanding with Language Models**
 Lang Cao. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.19378)]

**Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks**

*Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang. arxiv 2025.* [[pdf](https://arxiv.org/pdf/2503.05445)]

**CoddLLM: Empowering Large Language Models for Data Analytics**

*Jiani Zhang, Hengrui Zhang, Rishav Chakravarti, Yiqun Hu, Patrick Ng, Asterios Katsifodimos, Huzefa Rangwala, George Karypis, Alon Halevy. arxiv 2025.* [[pdf](https://arxiv.org/pdf/2502.00329)]

**Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks**

*Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang. arxiv 2025.* [[pdf](https://arxiv.org/pdf/2503.05445)]

**Generating highly customizable python code for data processing with large language models**

*Immanuel Trummer. VLDB Journal 2025.* [[pdf](https://link.springer.com/article/10.1007/s00778-025-00900-4)]

**The Dawn of Natural Language to SQL: Are We Fully Ready?**

*Boyan Li, Yuyu Luo, Chengliang Chai, Guoliang Li, Nan Tang. VLDB 2024.* [[pdf](https://arxiv.org/pdf/2406.01265)]

**PURPLE: Making a Large Language Model a Better SQL Writer**

*Ren, Tonghui, Yuankai Fan, Zhenying He, Ren Huang, Jiaqi Dai, Can Huang, Yinan Jing, Kai Zhang, Yifan Yang, and X. Sean Wang. ICDE 2024.* [[pdf](https://arxiv.org/pdf/2403.20014)]

**SM3-Text-to-Query: Synthetic Multi-Model Medical Text-to-Query Benchmark**

*Sithursan Sivasubramaniam, Cedric Osei-Akoto, Yi Zhang, Kurt Stockinger, Jonathan Fuerst. NeurIPS 2024.* [[pdf](https://arxiv.org/pdf/2411.05521)]

**Towards Automated Cross-domain Exploratory Data Analysis through Large Language Models**

*Jun-Peng Zhu, Boyan Niu, Peng Cai, Zheming Ni, Jianwei Wan, Kai Xu, Jiajun Huang, Shengbo Ma, Bing Wang, Xuan Zhou, Guanglei Bao, Donghui Zhang, Liu Tang, and Qi Liu. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2412.07214)]

**Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows**

*Fangyu Lei, Jixuan Chen, Yuxiao Ye, Ruisheng Cao, Dongchan Shin, Hongjin Su, Zhaoqing Suo, Hongcheng Gao, Wenjing Hu, Pengcheng Yin, Victor Zhong, Caiming Xiong, Ruoxi Sun, Qian Liu, Sida Wang, Tao Yu. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2411.07763)]

**SiriusBI: Building End-to-End Business Intelligence Enhanced by Large Language Models**

*Jie Jiang, Haining Xie, Yu Shen, Zihan Zhang, Meng Lei, Yifeng Zheng, Yide Fang, Chunyou Li, Danqing Huang, Wentao Zhang, Yang Li, Xiaofeng Yang, Bin Cui, Peng Chen. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2411.06102)]

**Grounding Natural Language to SQL Translation with Data-Based Self-Explanations**

*Yuankai Fan, Tonghui Ren, Can Huang, Zhenying He, X. Sean Wang. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2411.02948)]

**LR-SQL: A Supervised Fine-Tuning Method for Text2SQL Tasks under Low-Resource Scenarios**

*Wen Wuzhenghong, Zhang Yongpan, Pan Su, Sun Yuwei, Lu Pengwei, Ding Cheng. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2410.11457)]

**CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL**

*Mohammadreza Pourreza, Hailong Li, Ruoxi Sun, Yeounoh Chung, Shayan Talaei, Gaurav Tarlok Kakkar, Yu Gan, Amin Saberi, Fatma Ozcan, Sercan O. Arik. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2410.01943)]

**MoMQ: Mixture-of-Experts Enhances Multi-Dialect Query Generation across Relational and Non-Relational Databases**

*Zhisheng Lin, Yifu Liu, Zhiling Luo, Jinyang Gao, Yu Li. arxiv 2024.* [[pdf](https://arxiv.org/pdf/2410.18406)]

**Data Interpreter: An LLM Agent For Data Science**
 Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, et al. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2402.18679)]

**Contextualized Data-Wrangling Code Generation in Computational Notebooks**
 Junjie Huang, Daya Guo, Chenglong Wang, Jiazhen Gu, Shuai Lu, Jeevana Priya Inala, Cong Yan, Jianfeng Gao, Nan Duan, Michael R. Lyu. *ASE 2024.* [[pdf](https://doi.org/10.48550/arXiv.2409.13551)]

**CodeS: Towards Building Open-source Language Models for Text-to-SQL**
 Haoyang Li, Jing Zhang, Hanbing Liu, Ju Fan, Xiaokang Zhang, Jun Zhu, Renjie Wei, Hongyan Pan, Cuiping Li, Hong Chen. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3654930)]

**PET-SQL: A Prompt-Enhanced Two-Round Refinement of Text-to-SQL with Cross-consistency**
 Zhishuai Li, Xiang Wang, Jingjing Zhao, Sun Yang, Guoqing Du, Xiaoru Hu, Bin Zhang, Yuxiao Ye, Ziyue Li, Rui Zhao, Hangyu Mao. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.09732)]

**Table-GPT: A Large Multimodal Model with Tabular Data Integration**
 Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle Rifinski Fainman, Dongmei Zhang, Surajit Chaudhuri. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3654979)]

**Improved Baselines with Visual Instruction Tuning**
 Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee. *CVPR 2024.* [[pdf](https://ieeexplore.ieee.org/document/10655294)]

**CABINET: Content Relevance based Noise Reduction for Table Question Answering**
 Sohan Patnaik, Heril Changwal, Milan Aggarwal, Sumit Bhatia, Yaman Kumar, Balaji Krishnamurthy. *ICLR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.01155)]

**TableGPT2: A Large Multimodal Model with Tabular Data Integration**
 Aofeng Su, Aowen Wang, Chao Ye, Chen Zhou, et al. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2411.02059)]

**CHESS: Contextual Harnessing for Efficient SQL Synthesis**
 Shayan Talaei, Mohammadreza Pourreza, Yu-Chen Chang, Azalia Mirhoseini, Amin Saberi. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.16755)]

**Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding**
 Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee, Tomas Pfister. *ICLR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.04398)]

**FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis**
 Chao Zhang, Yuren Mao, Yijiang Fan, Yu Mi, Yunjun Gao, Lu Chen, Dongfang Lou, Jinshu Lin. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3626246.3653375)]

**ReAcTable: Enhancing ReAct for Table Question Answering**
 Yunjia Zhang, Jordan Henkel, Avrilia Floratou, Joyce Cahoon, Shaleen Deep, Jignesh M. Patel. *VLDB 2024.* [[pdf](https://doi.org/10.14778/3659437.3659452)]

**TabPedia: Towards Comprehensive Visual Table Understanding with Concept Synergy**
 Weichao Zhao, Hao Feng, Qi Liu, Jingqun Tang, Shu Wei, Binghong Wu, Lei Liao, Yongjie Ye, Hao Liu, Wengang Zhou, Houqiang Li, Can Huang. *NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2406.01326)]

**TaPERA: Enhancing Faithfulness and Interpretability in Long-Form Table QA by Content Planning and Execution-based Reasoning**
 Yilun Zhao, Lyuhao Chen, Arman Cohan, Chen Zhao. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.692/)]

**Multimodal Table Understanding**
 Mingyu Zheng, Xinwei Feng, Qingyi Si, Qiaoqiao She, Zheng Lin, Wenbin Jiang, Weiping Wang. *ACL 2024.* [[pdf](https://doi.org/10.48550/arXiv.2406.08100)]

**TAT-LLM: A Specialized Language Model for Discrete Reasoning over Financial Tabular and Textual Data**
 Fengbin Zhu, Ziyang Liu, Fuli Feng, Chao Wang, Moxin Li, Tat Seng Chua. *ICAIF 2024.* [[pdf](https://doi.org/10.1145/3677052.3698685)]

**Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments**
 Sitao Cheng, Ziyuan Zhuang, Yong Xu, Fangkai Yang, Chaoyun Zhang, Xiaoting Qin, Xiang Huang, Ling Chen, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, Qi Zhang. *ACL 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.08593)]

**FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering** 

Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, Jianyong Wang. *AAAI 2024.* [[pdf](https://doi.org/10.48550/arXiv.2308.12060)]

**NAT-NL2GQL: A Novel Multi-Agent Framework for Translating Natural Language to Graph Query Language** 

Yuanyuan Liang, Tingyu Xie, Gan Peng, Zihao Huang, Yunshi Lan, Weining Qian. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2412.10434)]

**GraphGPT: Graph Instruction Tuning for Large Language Models** 

Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, Chao Huang. *SIGIR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2310.13023)]

**InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment** 

Jianing Wang, Junda Wu, Yupeng Hou, Yao Liu, Ming Gao, Julian McAuley. *ACL 2024.* [[pdf](https://aclanthology.org/2024.findings-acl.801/)]

**Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models** 

Guanming Xiong, Junwei Bao, Wen Zhao. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.569/)]

**Language is All a Graph Needs** 

Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, Yongfeng Zhang. *EACL 2024.* [[pdf](https://aclanthology.org/2024.findings-eacl.132/)]

**R3-NL2GQL: A Model Coordination and Knowledge Graph Alignment Approach for NL2GQL** 

Yuhang Zhou, Yu He, Siyu Tian, Yuchen Ni, Zhangyue Yin, Xiang Liu, Chuanjun Ji, Sen Liu, Xipeng Qiu, Guangnan Ye, Hongfeng Chai. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.findings-emnlp.800/)]

**Direct Preference Optimization: Your Language Model is Secretly a Reward Model** 

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn. *NeurIPS 2023.* [[pdf](https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)]

**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** 

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. *NeurIPS 2023.* [[pdf](https://doi.org/10.48550/arXiv.2306.05685)]

**PaLM: Scaling Language Modeling with Pathways** 

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al. *JMLR 2023.* [[pdf](https://dl.acm.org/doi/10.5555/3648699.3648939)]

**S3HQA: A Three-Stage Approach for Multi-hop Text-Table Hybrid Question Answering** 

Fangyu Lei, Xiang Li, Yifan Wei, Shizhu He, Yiming Huang, Jun Zhao, Kang Liu. *ACL 2023.* [[pdf](https://aclanthology.org/2023.acl-short.147/)]

**Natural Language to Code Generation in Interactive Data Science Notebooks** 

Pengcheng Yin, Wen-Ding Li, Kefan Xiao, Abhishek Rao, Yeming Wen, Kensen Shi, Joshua Howland, Paige Bailey, Michele Catasta, Henryk Michalewski, Oleksandr Polozov, Charles Sutton. *ACL 2023.* [[pdf](https://aclanthology.org/2023.acl-long.9/)]

**StructGPT: A General Framework for Large Language Model to Reason over Structured Data** 

Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, Ji-Rong Wen. *EMNLP 2023.* [[pdf](https://doi.org/10.48550/arXiv.2305.09645)]

**Few-shot Text-to-SQL Translation using Structure and Content Prompt Learning**

*Zihui Gu, Ju Fan, Nan Tang, et al. SIGMOD 2023.* [[pdf](http://iir.ruc.edu.cn/~fanj/papers/sigmod2023-scprompt.pdf)]

**UniKGQA: Unified Retrieval and Reasoning for Solving Multi-hop Question Answering Over Knowledge Graph** 

Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen. *ICLR 2023.* [[pdf](https://doi.org/10.48550/arXiv.2212.00959)]

**From BERT to GPT-3 Codex: Harnessing the Potential of Very Large Language Models for Data Management**

*Immanuel Trummer. VLDB 2022.* [[pdf](https://www.vldb.org/pvldb/vol15/p3770-trummer.pdf)]

**Subgraph Retrieval Enhanced Model for Multi-hop Knowledge Base Question Answering** 

Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, Hong Chen. *ACL 2022.* [[pdf](https://aclanthology.org/2022.acl-long.396/)]

**Inductive representation learning on large graphs** 

William L. Hamilton, Rex Ying, Jure Leskovec. *NeurIPS 2017.* [[pdf](https://dl.acm.org/doi/10.5555/3294771.3294869)]

**Semi-Supervised Classification with Graph Convolutional Networks** 

Thomas N. Kipf, Max Welling. *ICLR 2017.* [[pdf](https://doi.org/10.48550/arXiv.1609.02907)]

**A Comparison of Current Graph Database Models** 

Renzo Angles. *ICDEW 2012.* [[pdf](https://doi.org/10.1109/ICDEW.2012.31)]

**A Relational Model of Data for Large Shared Data Banks** 

E. F. Codd. *Communications of the ACM 1970.* [[pdf](https://doi.org/10.1145/362384.362685)]


### 5.2 Semi-Structured Data Analysis

**MiMoTable: A Multi-scale Spreadsheet Benchmark with Meta Operations for Table Reasoning**
 Zheng Li, Yang Du, Mao Zheng, Mingyang Song. *COLING 2025.* [[pdf](https://doi.org/10.48550/arXiv.2412.11711)]

**SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation**
 Zeyao Ma, Bohan Zhang, Jing Zhang, Jifan Yu, Xiaokang Zhang, Xiaohan Zhang, Sijia Luo, Xi Wang, Jie Tang. *NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2406.14991)]

**TempTabQA: Temporal Question Answering for Semi-Structured Tables**
 Vivek Gupta, Pranshu Kandoi, Mahek Bhavesh Vora, Shuo Zhang, Yujie He, Ridho Reinanda, Vivek Srikumar. *EMNLP 2023.* [[pdf](https://doi.org/10.48550/arXiv.2311.08002)]

**Querying Semi-Structured Data**
 Serge Abiteboul. *ICDT 1997.* [[pdf](https://dl.acm.org/doi/10.5555/645502.656103)]

### 5.3 Unstructured Data Analysis

**DAgent: A Relational Database-Driven Data Analysis Report Generation Agent**
Wenyi Xu, Yuren Mao, Xiaolu Zhang, Chao Zhang, Xuemei Dong, Mengfei Zhang, Jun Zhou, Yunjun Gao. *arxiv 2025.* [[pdf](https://arxiv.org/pdf/2503.13269)]

**DocFormerv2: Local Features for Document Understanding**
 Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, R. Manmatha. *AAAI 2024.* [[pdf](https://doi.org/10.1609/aaai.v38i2.27828)]

**Improving Code Summarization With Tree Transformer Enhanced by Position-Related Syntax Complement**
 Jie Song, Zexin Zhang, Zirui Tang, Shi Feng, Yu Gu. *IEEE TAI 2024.* [[pdf](https://ieeexplore.ieee.org/document/10510878/metrics#metrics)]

**Repoformer: Selective Retrieval for Repository-Level Code Completion**
 Di Wu, Wasi Uddin Ahmad, Dejiao Zhang, Murali Krishna Ramanathan, Xiaofei Ma. *ICML 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.10059)]

**Large Language Model for Vulnerability Detection: Emerging Results and Future Directions**
 Xin Zhou, Ting Zhang, David Lo. *ICSE-NIER 2024.* [[pdf](https://doi.org/10.1145/3639476.3639762)]

**DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain for Versatile Document Understanding**
 Hao Feng, Qi Liu, Hao Liu, Jingqun Tang, Wengang Zhou, Houqiang Li, Can Huang. *SCIS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2311.11810)]

**Focus Anywhere for Fine-grained Multi-page Document Understanding**
 Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, Xiangyu Zhang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2405.14295)]

**mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding**
 Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Ji Zhang, Qin Jin, Fei Huang, Jingren Zhou. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.findings-emnlp.175/)]

**General OCR Theory: Towards OCR - 2.0 via a Unified End - to - end Model**
 Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, Chunrui Han, Xiangyu Zhang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2409.01704v1)]

**Automatic Semantic Augmentation of Language Model Prompts (for Code Summarization)**
 Toufique Ahmed, Kunal Suresh Pai, Premkumar Devanbu, Earl Barr. *ICSE 2024.* [[pdf](https://doi.org/10.1145/3597503.3639183)]

**CoCoMIC: Code Completion by Jointly Modeling In-file and Cross-file Context**
 Yangruibo Ding, Zijian Wang, Wasi Ahmad, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang. *LREC-COLING 2024.* [[pdf](https://aclanthology.org/2024.lrec-main.305/)]

**Large Language Models are Few-Shot Summarizers: Multi-Intent Comment Generation via In-Context Learning**
 Mingyang Geng, Shangwen Wang, Dezun Dong, Haotian Wang, Ge Li, Zhi Jin, Xiaoguang Mao, Xiangke Liao. *ICSE 2024.* [[pdf](https://doi.org/10.1145/3597503.3608134)]

**Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks**
 Zhongxin Liu, Zhijie Tang, Junwei Zhang, Xin Xia, Xiaohu Yang. *ICSE 2024.* [[pdf](https://doi.org/10.1145/3597503.3639142)]

**SCLA: Automated Smart Contract Summarization via LLMs and Semantic Augmentation**
 Yingjie Mao, Xiaoqi Li, Wenkai Li, Xin Wang, Lei Xie. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.04863)]

**Software Vulnerability Detection with GPT and In-Context Learning**
 Zhihong Liu, Qing Liao, Wenchao Gu, Cuiyun Gao. *DSC 2023.* [[pdf](https://ieeexplore.ieee.org/abstract/document/10381286)]

**Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding**
 Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova. *ICML 2023.* [[pdf](https://doi.org/10.48550/arXiv.2210.03347)]

**DUBLIN: Visual Document Understanding By Language-Image Network**
 Kriti Aggarwal, Aditi Khandelwal, Kumar Tanmay, Owais Khan Mohammed, Qiang Liu, Monojit Choudhury, Hardik Chauhan, Subhojit Som, Vishrav Chaudhary, Saurabh Tiwary. *EMNLP 2023.* [[pdf](https://aclanthology.org/2023.emnlp-industry.65/)]

**Code Structure–Guided Transformer for Source Code Summarization**
 Shuzheng Gao, Cuiyun Gao, Yulan He, Jichuan Zeng, Lunyiu Nie, Xin Xia, Michael Lyu. *ACM Transactions on Software Engineering and Methodology 2023.* [[pdf](https://doi.org/10.1145/3522674)]

**RepoFusion: Training Code Models to Understand Your Repository**
 Disha Shrivastava, Denis Kocetkov, Harm de Vries, Dzmitry Bahdanau, Torsten Scholak. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2306.10998)]

**Vulnerability Detection by Learning From Syntax-Based Execution Paths of Code**
 Junwei Zhang, Zhongxin Liu, Xing Hu, Xin Xia, Shanping Li. *IEEE TSE 2023.* [[pdf](https://ieeexplore.ieee.org/document/10153647)]

**Unifying Vision, Text, and Layout for Universal Document Processing**
 Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, Mohit Bansal. *CVPR 2023.* [[pdf](https://arxiv.org/abs/2212.02623v3)]

**CodeBERT: A Pre-Trained Model for Programming and Natural Languages**
 Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, Ming Zhou. *EMNLP 2020.* [[pdf](https://doi.org/10.48550/arXiv.2002.08155)]

**Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**
 Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. *JMLR 2020.* [[pdf](https://dl.acm.org/doi/10.5555/3455716.3455856)]

**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
 Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. *arXiv 2020.* [[pdf](https://arxiv.org/abs/2010.11929)]

**The Probabilistic Relevance Framework: BM25 and Beyond**
 Stephen Robertson, Hugo Zaragoza. *FNTIR 2009.* [[pdf](https://dl.acm.org/doi/10.1561/1500000019)]

**The JPEG Still Picture Compression Standard**
 Gregory K. Wallace. *Communications of the ACM 1991.* [[pdf](https://doi.org/10.1145/103085.103089)]


### 5.4 Data Exploration

**AutoDDG: Automated Dataset Description Generation using Large Language Models**

*Haoxiang Zhang, Yurong Liu, Wei-Lun (Allen) Hung, Aécio Santos, Juliana Freire. arxiv 2025.* [[pdf](https://arxiv.org/pdf/2502.01050)]

**Db-gpt: Empowering database interactions with private large language models**

*Siqiao Xue, Caigao Jiang, Wenhui Shi, Fangyin Cheng, et al. arxiv 2023.* [[pdf](https://arxiv.org/pdf/2312.17449)]

### 5.5 Data Visualization

**LLM4Vis: Explainable Visualization Recommendation using ChatGPT**

*Lei Wang, Songheng Zhang, Yun Wang, Ee-Peng Lim, Yong Wang. EMNLP 2023.* [[pdf](https://aclanthology.org/2023.emnlp-industry.64.pdf)]


## 6. LLM for Data System Optimization

### 6.1 Configuration Tuning

**Automatic Database Configuration Debugging using Retrieval-Augmented Language Models**

Sibei Chen, Ju Fan, Bin Wu, Nan Tang, Chao Deng, Pengyi Wang, Ye Li, Jian Tan, Feifei Li, Jingren Zhou, Xiaoyong Du. *SIGMOD 2025.* [[pdf](https://arxiv.org/pdf/2412.07548)]

**λ-Tune: Harnessing Large Language Models for Automated Database System Tuning**

Victor Giannankouris, Immanuel Trummer. *SIGMOD 2025.* [[pdf](https://doi.org/10.48550/arXiv.2411.03500)]

**E2ETune: End-to-End Knob Tuning via Fine-tuned Generative Language Model**

Xinmei Huang, Haoyang Li, Jing Zhang, Xinxin Zhao, Zhiming Yao, Yiyan Li, Tieying Zhang, Jianjun Chen, Hong Chen, Cuiping Li. *VLDB 2025.* [[pdf](https://doi.org/10.48550/arXiv.2404.11581)]

**LLMIdxAdvis: Resource-Efficient Index Advisor Utilizing Large Language Model**

Xinxin Zhao, Haoyang Li, Jing Zhang, Xinmei Huang, Tieying Zhang, Jianjun Chen, Rui Shi, Cuiping Li, Hong Chen. *arxiv 2025.* [[pdf](https://arxiv.org/pdf/2503.07884)]

**LATuner: An LLM-Enhanced Database Tuning System Based on Adaptive Surrogate Model**

Chongjiong Fan, Zhicheng Pan, Wenwen Sun, Chengcheng Yang, Wei-Neng Chen. *ECML PKDD 2024.* [[pdf](https://doi.org/10.1007/978-3-031-70362-1_22)]

**GPTuner: A Manual-Reading Database Tuning System via GPT-Guided Bayesian Optimization**

Jiale Lao, Yibo Wang, Yufei Li, Jianping Wang, Yunjia Zhang, Zhiyuan Cheng, Wanghu Chen, Mingjie Tang, Jianguo Wang. *VLDB 2024.* [[pdf](https://doi.org/10.14778/3659437.3659449)]

**Is Large Language Model Good at Database Knob Tuning? A Comprehensive Experimental Evaluation**

Yiyan Li, Haoyang Li, Zhao Pu, Jing Zhang, Xinyi Zhang, Tao Ji, Luming Sun, Cuiping Li, Hong Chen. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2408.02213)]

**DB-GPT: Large Language Model Meets Database**

Xuanhe Zhou, Zhaoyan Sun, Guoliang Li. *Data Science and Engineering 2024.* [[pdf](https://link.springer.com/article/10.1007/s41019-023-00235-6)]

**DB-BERT: A Database Tuning Tool that "Reads the Manual"**

Immanuel Trummer. *SIGMOD 2022.* [[pdf](https://doi.org/10.1145/3514221.3517843)]

**Automatic Database Management System Tuning Through Large-scale Machine Learning**

Dana Van Aken, Andrew Pavlo, Geoffrey J. Gordon, Bohan Zhang. *SIGMOD 2017.* [[pdf](https://doi.org/10.1145/3035918.3064029)]

### 6.2 Query Optimization

**Can Large Language Models Be Query Optimizer for Relational Databases?**

Jie Tan, Kangfei Zhao, Rui Li, Jeffrey Xu Yu, Chengzhi Piao, Hong Cheng, Helen Meng, Deli Zhao, Yu Rong. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.05562)]

**A Query Optimization Method Utilizing Large Language Models**

*Zhiming Yao, Haoyang Li, Jing Zhang, Cuiping Li, Hong Chen. arxiv 2025.* [[pdf](https://arxiv.org/pdf/2503.06902)]

**Query Rewriting via LLMs**

Sriram Dharwada, Himanshu Devrani, Jayant Haritsa, Harish Doraiswamy. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.12918)]

**LLM-R2: A Large Language Model Enhanced Rule-Based Rewrite System for Boosting Query Efficiency**

Zhaodonghui Li, Haitao Yuan, Huiming Wang, Gao Cong, Lidong Bing. *VLDB 2024.* [[pdf](https://doi.org/10.14778/3696435.3696440)]

**R-Bot: An LLM-based Query Rewrite System**

Zhaoyan Sun, Xuanhe Zhou, Guoliang Li. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2412.01661)]

**Query Rewriting via Large Language Models**

Jie Liu, Barzan Mozafari. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.09060)]

**The Unreasonable Effectiveness of LLMs for Query Optimization**

Peter Akioyamen, Zixuan Yi, Ryan Marcus. *NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2411.02862)]

### 6.3 Anomaly Diagnosis

**Query Performance Explanation through Large Language Model for HTAP Systems**  

Haibo Xiu, Li Zhang, Tieying Zhang, Jun Yang, Jianjun Chen. *arxiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2412.01709)]

**D-Bot: Database Diagnosis System using Large Language Models**

Xuanhe Zhou, Guoliang Li, Zhaoyan Sun, Zhiyuan Liu, Weize Chen, Jianming Wu, Jiesi Liu, Ruohang Feng, Guoyang Zeng. *VLDB 2024.* [[pdf](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbot_vldb_camera_ready_v1.pdf)]

**Panda: Performance Debugging for Databases using LLM Agents**

Vikramank Singh, Kapil Eknath Vaidya, Vinayshekhar Bannihatti Kumar, Sopan Khosla, Balakrishnan Narayanaswamy, Rashmi Gangadharaiah, Tim Kraska. *CIDR 2024.* [[pdf](https://www.cidrdb.org/cidr2024/papers/p6-singh.pdf)]

**DBG-PT: A Large Language Model Assisted Query Performance Regression Debugger**

Victor Giannakouris, Immanuel Trummer. *VLDB 2024.* [[pdf](https://doi.org/10.14778/3685800.3685869)]

**LLM As DBA**  

Xuanhe Zhou, Guoliang Li, Zhiyuan Liu. *arXiv 2023.* [[pdf](https://arxiv.org/pdf/2308.05481)]
