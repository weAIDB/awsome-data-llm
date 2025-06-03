# A Survey of LLM × DATA



> A collection of papers and projects related to LLMs and corresponding data-centric methods. [![arXiv](https://camo.githubusercontent.com/dc1f84975e5d05724930d5c650e4b6eaea49e9f4c03d00de50bd7bf950394b4f/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f6261646765732f7261772f6d61696e2f70617065722d706167652d736d2d6461726b2e737667)](https://arxiv.org/abs/2505.18458v2)
>
> If you find our survey useful, please cite the paper:

```
@article{LLMDATASurvey,
    title={A Survey of LLM × DATA},
    author={Xuanhe Zhou, Junxuan He, Wei Zhou, Haodong Chen, Zirui Tang, Haoyu Zhao, Xin Tong, Guoliang Li, Youmin Chen, Jun Zhou, Zhaojun Sun, Binyuan Hui, Shuo Wang, Conghui He, Zhiyuan Liu, Jingren Zhou, Fan Wu},
    year={2025},
    journal={arXiv preprint arXiv:2505.18458v3},
    url={https://arxiv.org/abs/2505.18458v3}
}
```



## 🌤 The IaaS Concept of DATA4LLM



The **IaaS** concept for LLM data (phonetically echoing *Infrastructure as a Service*) defines the characteristics of high-quality datasets along four key dimensions: (1) **Inclusiveness** ensures broad coverage across domains, tasks, sources, languages, styles, and modalities. (2) **Abundance** emphasizes sufficient and well-balanced data volume to support scaling, fine-tuning, and continual learning without overfitting. (3) **Articulation** requires clear, coherent, and instructive content with step-by-step reasoning to enhance model understanding and task performance. (4) **Sanitization** involves rigorous filtering to remove private, toxic, unethical, and misleading content, ensuring data safety, neutrality, and compliance. [![arXiv](https://camo.githubusercontent.com/dc1f84975e5d05724930d5c650e4b6eaea49e9f4c03d00de50bd7bf950394b4f/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f6261646765732f7261772f6d61696e2f70617065722d706167652d736d2d6461726b2e737667)](https://github.com/SUPERZJ827/LLM4DB/blob/main/assets/data_llm_survey_v3.pdf)

[![Cover](https://github.com/SUPERZJ827/LLM4DB/raw/main/assets/iaas_overview_v2.png)](https://github.com/SUPERZJ827/LLM4DB/blob/main/assets/iaas_overview_v2.png)



## Table of Contents

- [Datasets](#datasets)
- [0 Data Characteristics across LLM Stages](#0-data-characteristics-across-llm-stages)
  - [Data for Pretraining](#data-for-pretraining)
  - [Data for Continual Pre-training](#data-for-continual-pre-training)
  - [Data for Supervised Fine-Tuning (SFT)](#data-for-supervised-fine-tuning-sft)
  - [Data for Reinforcement Learning (RL)](#data-for-reinforcement-learning-rl)
  - [Data for Retrieval-Augmented Generation (RAG)](#data-for-retrieval-augmented-generation-rag)
  - [Data for LLM Evaluation](#data-for-llm-evaluation)
  - [Data for LLM Agents](#data-for-llm-agents)
- [1 Data Processing for LLM](#1-data-processing-for-llm)
  - [1.1 Data Acquisition](#11-data-acquisition)
  - [1.2 Data Deduplication](#12-data-deduplication)
  - [1.3 Data Filtering](#13-data-filtering)
  - [1.4 Data Selection](#14-data-selection)
  - [1.5 Data Mixing](#15-data-mixing)
  - [1.6 Data Distillation and Synthesis](#16-data-distillation-and-synthesis)
  - [1.7 End-to-End Data Processing Pipelines](#17-end-to-end-data-processing-pipelines)
- [2 Data Storage for LLM](#2-data-storage-for-llm)
  - [2.1 Data Formats](#21-data-formats)
  - [2.2 Data Distribution](#22-data-distribution)
  - [2.3 Data Organization](#23-data-organization)
  - [2.4 Data Movement](#24-data-movement)
  - [2.5 Data Fault Tolerance](#25-data-fault-tolerance)
  - [2.6 KV Cache](#26-kv-cache)
- [3 Data Serving for LLM](#3-data-serving-for-llm)
  - [3.1 Data Shuffling](#31-data-shuffling)
  - [3.2 Data Compression](#32-data-compression)
  - [3.3 Data Packing](#33-data-packing)
  - [3.4 Data Provenance](#34-data-provenance)
- [4 LLM for Data Management](#4-llm-for-data-management)
  - [4.1 LLM for Data Manipulation](#41-llm-for-data-manipulation)
  - [4.2 LLM for Data Analysis](#42-llm-for-data-analysis)
  - [4.3 LLM for Data System Optimization](#43-llm-for-data-system-optimization)
- [5 Challenges and Future Directions](#5-challenges-and-future-directions)
  - [5.1 Data Management for LLM](#51-data-management-for-llm)
  - [5.2 LLM for Data Management](#52-llm-for-data-management)



## Datasets

1. CommonCrawl: [[source](https://commoncrawl.org/latest-crawl)]

1. The Stack: [[HuggingFace](https://huggingface.co/datasets/bigcode/the-stack-v2)]

1. RedPajama: [[Github](https://github.com/togethercomputer/RedPajama-Data)]

1. SlimPajama-627B-DC: [[HuggingFace](https://huggingface.co/datasets/MBZUAI-LLM/SlimPajama-627B-DC)]

1. Alpaca-CoT: [[Github](https://github.com/PhoebusSi/Alpaca-CoT?tab=readme-ov-file)]

1. LLaVA-Pretrain: [[HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)]

1. Wikipedia: [[HuggingFace](https://huggingface.co/datasets/wikimedia/wikipedia)]

1. C4: [[HuggingFace](https://huggingface.co/datasets/allenai/c4)]

1. BookCorpus: [[HuggingFace](https://huggingface.co/datasets/bookcorpus/bookcorpus)]

1. Arxiv: [[HuggingFace](https://huggingface.co/datasets/arxiv-community/arxiv_dataset)]

1. PubMed: [[source](https://pubmed.ncbi.nlm.nih.gov/download/)]

1. StackExchange: [[source](https://archive.org/details/stackexchange)]

1. OpenWebText2: [[source](https://openwebtext2.readthedocs.io/en/latest/)]

1. OpenWebMath: [[HuggingFace](https://huggingface.co/datasets/open-web-math/open-web-math)]

1. Falcon-RefinedWeb: [[HuggingFace](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)]

1. CCI 3.0: [[HuggingFace](https://huggingface.co/datasets/BAAI/CCI3-Data)]

1. OmniCorpus: [[Github](https://github.com/OpenGVLab/OmniCorpus?tab=readme-ov-file)]

1. WanJuan3.0: [[source](https://opendatalab.org.cn/OpenDataLab/WanJuan3)]

   

## 0 Data Characteristics across LLM Stages

[**⬆️top**](#table-of-contents)

### Data for Pretraining

1. OBELICS:  **OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents**  
   Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh. *NeurIPS 2023*. [[pdf](https://neurips.cc/virtual/2023/poster/73589 )] 
2. Aligning Books and Movies: **Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books**  
   Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, Sanja Fidler. *ICCV 2015*.[[pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhu_Aligning_Books_and_ICCV_2015_paper.pdf)] 

### Data for Continual Pre-training

1. MedicalGPT: **MedicalGPT: Training Medical GPT Model**   
   Ming Xu. [[Github](https://github.com/shibing624/MedicalGPT)]
2. BBT-FinCorpus: **BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark**   
   Dakuan Lu, Hengkui Wu, Jiaqing Liang, Yipei Xu, Qianyu He, Yipeng Geng, Mengkun Han, Yingsi Xin, Yanghua Xiao. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2302.09432 )] 

### Data for Supervised Fine-Tuning (SFT)

#### General Instruction Following

1. Databricks-dolly-15K:  **Free dolly: Introducing the world’s first truly open instruction-tuned llm**  

   Mike Conover and Matt Hayes and Ankit Mathur and Jianwei Xie and Jun Wan and Sam Shah and Ali Ghodsi and Patrick Wendell and Matei Zaharia and Reynold Xin. [[source](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)]

#### Specific Domain Usage

1. MedicalGPT: **MedicalGPT: Training Medical GPT Model**   
   Ming Xu. [[Github](https://github.com/shibing624/MedicalGPT)]
2. DISC-LawSFT: **DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services**  
   Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, Zhongyu Wei. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2309.11325)]

### Data for Reinforcement Learning (RL)

#### RLHF

1. MedicalRLHF: **MedicalGPT: Training Medical GPT Model**    
   Ming Xu. [[Github](https://github.com/shibing624/MedicalGPT)]
2. UltraFeedback: **UltraFeedback: Boosting Language Models with Scaled AI Feedback**  
   Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, Maosong Sun. *ICML 2024*. [[pdf](https://arxiv.org/abs/2310.01377)]

#### RoRL

1. Group Relative Policy Optimization (GRPO):  **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**  
   DeepSeek-AI. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2501.12948)]
2. long-CoT RL: **Kimi k1.5: Scaling Reinforcement Learning with LLMs**  
   Kimi Team. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2501.12599)]

### Data for Retrieval-Augmented Generation (RAG)

1. DH-RAG: **DH-RAG: A Dynamic Historical Context-Powered Retrieval-Augmented Generation Method for Multi-Turn Dialogue**  
   Feiyuan Zhang, Dezhi Zhu, James Ming, Yilun Jin, Di Chai, Liu Yang, Han Tian, Zhaoxin Fan, Kai Chen. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2502.13847)]
2. Medical Graph RAG:**Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation**  
   Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, Vicente Grau. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2408.04187)]
3. ERAGent: **ERAGent: Enhancing Retrieval-Augmented Language Models with Improved Accuracy, Efficiency, and Personalization**  
   Yunxiao Shi, Xing Zi, Zijing Shi, Haimin Zhang, Qiang Wu, Min Xu. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2405.06683)]
4. PersonaRAG: **PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents**  
   Saber Zerhoudi, Michael Granitzer. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2407.09394)]
5. DISC-LawLLM: **DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services**  
   Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, Zhongyu Wei. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2309.11325)]

### Data for LLM Evaluation

1. MMMU: **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI**  
   Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen. *CVPR 2024*. [[pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.pdf)]
2. LexEval: **LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models**  
   Haitao Li, You Chen, Qingyao Ai, Yueyue Wu, Ruizhe Zhang, Yiqun Liu. *NeurIPS 2024*. [[pdf](https://arxiv.org/abs/2409.20288)]
3. MedQA: **What disease does this patient have? a large-scale open domain question answering dataset from medical exams**   
   Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, Peter Szolovits. *AAAI 2021*. [[pdf](https://arxiv.org/abs/2009.13081)]
4. **Evaluating Large Language Models Trained on Code**  
   Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba. *arXiv 2021*. [[pdf](https://arxiv.org/abs/2107.03374)]

### Data for LLM Agents

1. STeCa: **STeCa: Step-level Trajectory Calibration for LLM Agent Learning**  
   Hanlin Wang, Jian Wang, Chak Tou Leong, Wenjie Li. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2502.14276)]
2. **Large Language Model-Based Agents for Software Engineering: A Survey**  
   Junwei Liu, Kaixin Wang, Yixuan Chen, Xin Peng, Zhenpeng Chen, Lingming Zhang, Yiling Lou. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2409.02977)]
3. UltraInteract: **Advancing LLM Reasoning Generalists with Preference Trees**  
   Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, Maosong Sun. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2404.02078)]
4. AutoTools: **Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents**  
   Zhengliang Shi, Shen Gao, Lingyong Yan, Yue Feng, Xiuyi Chen, Zhumin Chen, Dawei Yin, Suzan Verberne, Zhaochun Ren. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2405.16533)]
5. UltraChat: **Enhancing Chat Language Models by Scaling High-quality Instructional Conversations**  
   Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, Bowen Zhou. *EMNLP 2023*. [[pdf](https://aclanthology.org/2023.emnlp-main.183/)]

## 1 Data Processing for LLM

[⬆️top](#table-of-contents)

### 1.1 Data Acquisition

#### Data Sources

##### Public Data

1. CommonCrawl: [[source](https://commoncrawl.org/)]
2. Project Gutenberg: [[source](https://www.gutenberg.org/)]
3. Open Library: [[source](https://openlibrary.org/)]
4. GitHub: [[source](https://github.com/)]
5. GitLab: [[source]( https://gitlab.com/)]
6. Bitbucket: [[source](https://bitbucket.org/product/)] 
7. CulturaX: **CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages**  
   Thuat Nguyen, Chien Van Nguyen, Viet Dac Lai, Hieu Man, Nghia Trung Ngo, Franck Dernoncourt, Ryan A. Rossi, Thien Huu Nguyen. *LREC-COLING 2024.* [[pdf](https://aclanthology.org/2024.lrec-main.377.pdf)]
8. The Stack: **The Stack: 3 TB of permissively licensed source code**  
   Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, Harm de Vries. *arXiv 2022*. [[pdf](https://arxiv.org/abs/2211.15533)]
9. mC4: **mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer**  
   Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel. *NAACL 2021.* [[pdf](https://aclanthology.org/2021.naacl-main.41.pdf)]
10. C4: **Exploring the limits of transfer learning with a unified text-to-text transformer**  
      Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. *JMLR 2020.* [[pdf](https://arxiv.org/abs/1910.10683)]
11. CodeSearchNet: **CodeSearchNet Challenge: Evaluating the State of Semantic Code Search**  
    Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, Marc Brockschmidt. *arXiv 2019*. [[pdf](https://arxiv.org/abs/1909.09436)]
12. BookCorpus: **Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books**  
    Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, Sanja Fidler. *ICCV 2015*.[[pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhu_Aligning_Books_and_ICCV_2015_paper.pdf)] 



#### Data Acquisition Methods

##### Website Crawling

1. Beautiful Soup: [[source](https://beautiful-soup-4.readthedocs.io/en/latest/)]
2. Selenium: [[Github]( https://github.com/seleniumhq/selenium)]
3. Playwright: [[source](https://playwright.dev/)]
4. Puppeteer: [[source](https://pptr.dev/)]
5. **An Empirical Comparison of Web Content Extraction Algorithms**  
   Janek Bevendorff, Sanket Gupta, Johannes Kiesel, Benno Stein. *SIGIR 2023*. [[pdf](https://dl.acm.org/doi/10.1145/3539618.3591920)]
6. Trafilatura: **Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction**  
   Adrien Barbaresi. *ACL 2021 Demo*. [[pdf](https://aclanthology.org/2021.acl-demo.15/)]
7. BET: **Fact or Fiction: Content Classification for Digital Libraries**  
   Aidan Finn, N. Kushmerick, Barry Smyth. *DELOS Workshops / Conferences 2001.* [[pdf](https://www.semanticscholar.org/paper/Fact-or-Fiction%3A-Content-Classification-for-Digital-Finn-Kushmerick/73ccd5c477b37a082f66557a1793852d405e4b6d)]

##### Layout Analysis

1. PaddleOCR: [[Github](https://github.com/paddlepaddle/paddleocr)]
2. YOLOv8: **YOLOv10: Real-Time End-to-End Object Detection**  
   Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding. *NeurIPS 2024.* [[pdf](https://arxiv.org/abs/2405.14458)]
3. UMIE: **UMIE: Unified Multimodal Information Extraction with Instruction Tuning**  
   Lin Sun, Kai Zhang, Qingyuan Li, Renze Lou. *AAAI 2024.* [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/29873)]
4. ChatEL: **ChatEL: Entity linking with chatbots**  
   Yifan Ding, Qingkai Zeng, Tim Weninger. *LREC | COLING 2024*. [[pdf](https://aclanthology.org/2024.lrec-main.275/)]
5. Vary:  **Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models**  
   Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, Xiangyu Zhang. *ECCV 2024.* [[pdf](https://link.springer.com/chapter/10.1007/978-3-031-73235-5_23)]
6. GOT2.0: **General OCR Theory: Towards OCR - 2.0 via a Unified End - to - end Model**  
   Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, Chunrui Han, Xiangyu Zhang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2409.01704v1)]
7. Fox:  **Focus Anywhere for Fine-grained Multi-page Document Understanding**  
   Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, Xiangyu Zhang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2405.14295)]
8. MinerU: **MinerU: An Open-Source Solution for Precise Document Content Extraction**  
   Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, Conghui He. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2409.18839)]
9. WebIE:  **WebIE: Faithful and Robust Information Extraction on the Web**  
   Chenxi Whitehouse, Clara Vania, Alham Fikri Aji, Christos Christodoulopoulos, Andrea Pierleoni. *ACL 2023.* [[pdf](https://aclanthology.org/2023.acl-long.428/)]
10. ReFinED: **ReFinED: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking**  
    Tom Ayoola, Shubhi Tyagi, Joseph Fisher, Christos Christodoulopoulos, Andrea Pierleoni. *NAACL 2022 Industry Track.* [[pdf](https://aclanthology.org/2022.naacl-industry.24.pdf)]
11. Alignment-Augmented Consistent Translation (AACTRANS) model:  **Alignment-Augmented Consistent Translation for Multilingual Open Information Extraction**  
    Keshav Kolluru, Muqeeth Mohammed, Shubham Mittal, Soumen Chakrabarti, Mausam. *ACL 2022.* [[pdf](https://aclanthology.org/2022.acl-long.179/)]
12. LayoutLMv3: **LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking**  
    Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei. *ACM Multimedia 2022.* [[pdf](https://arxiv.org/abs/2204.08387)]
13. CLIP-ViT:  **Learning Transferable Visual Models From Natural Language Supervision**  
    Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. *ICML 2021.* [[pdf](https://proceedings.mlr.press/v139/radford21a)]
14. Tesseract: **Tesseract: an open-source optical character recognition engine**  
    Anthony Kay. Linux Journal, Volume 2007. [[pdf](https://dl.acm.org/doi/10.5555/1288165.1288167)]



### 1.2 Data Deduplication

[⬆️top](#table-of-contents)

1. **Analysis of the Reasoning with Redundant Information Provided Ability of Large Language Models**  
   Wenbei Xie. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2310.04039v1)]
2. **Scaling Laws and Interpretability of Learning from Repeated Data**  
   Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Tom Henighan, Tristan Hume, Scott Johnston, Ben Mann, Chris Olah, Catherine Olsson, Dario Amodei, Nicholas Joseph, Jared Kaplan, Sam McCandlish. *arXiv 2022.* [[pdf](https://arxiv.org/abs/2205.10487)]

#### Exact Substring Matching

1. BaichuanSEED: **BaichuanSEED: Sharing the Potential of ExtensivE Data Collection and     Deduplication by Introducing a Competitive Large Language Model Baseline**    
   Guosheng Dong, Da Pan, Yiding Sun, Shusen Zhang, Zheng Liang, Xin Wu, Yanjun Shen, Fan Yang, Haoze Sun, Tianpeng Li, Mingan Lin, Jianhua Xu, Yufan Zhang, Xiaonan Nie, Lei Su, Bingning Wang, Wentao Zhang, Jiaxin Mao, Zenan Zhou, Weipeng Chen. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2408.15079)]
2. **Deduplicating Training Data Makes Language Models Better**    
   Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini. *ACL 2022.* [[pdf](https://arxiv.org/abs/2107.06499)]
3. Suffix arrays: **Suffix arrays: a new method for on-line string searches**  
   Udi Manber, Gene Myers. *SIAM Journal on Computing 1993.* [[pdf](https://doi.org/10.1137/0222058)]

#### Approximate Hashing-based Deduplication

1. BaichuanSEED: **BaichuanSEED: Sharing the Potential of ExtensivE Data Collection and     Deduplication by Introducing a Competitive Large Language Model Baseline**    
   Guosheng Dong, Da Pan, Yiding Sun, Shusen Zhang, Zheng Liang, Xin Wu, Yanjun Shen, Fan Yang, Haoze Sun, Tianpeng Li, Mingan Lin, Jianhua Xu, Yufan Zhang, Xiaonan Nie, Lei Su, Bingning Wang, Wentao Zhang, Jiaxin Mao, Zenan Zhou, Weipeng Chen. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2408.15079)]
2. LSHBloom: **LSHBloom: Memory-efficient, Extreme-scale Document Deduplication**  
   Arham Khan, Robert Underwood, Carlo Siebenschuh, Yadu Babuji, Aswathy Ajith, Kyle Hippe, Ozan Gokdemir, Alexander Brace, Kyle Chard, Ian Foster. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2411.04257)]
3. SimiSketch: **SimiSketch: Efficiently Estimating Similarity of streaming Multisets**   
   Fenghao Dong, Yang He, Yutong Liang, Zirui Liu, Yuhan Wu, Peiqing Chen, Tong Yang. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2405.19711)] 
4. DotHash: **DotHash: Estimating Set Similarity Metrics for Link Prediction and Document Deduplication**  
   Igor Nunes, Mike Heddes, Pere Vergés, Danny Abraham, Alex Veidenbaum, Alex Nicolau, Tony Givargis. *KDD 2023.* [[pdf](https://doi.org/10.1145/3580305.3599314)]
5. BPE Tokenization: **Formalizing BPE Tokenization**  
   Martin Berglund (Umeå University), Brink van der Merwe (Stellenbosch University). *NCMA 2023*. [[pdf](https://arxiv.org/abs/2309.08715)]
6. SlimPajama-DC: **SlimPajama-DC: Understanding Data Combinations for LLM Training**  
   Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Zhengzhong Liu, Hongyi Wang, Bowen Tan, Joel Hestness, Natalia Vassilieva, Daria Soboleva, Eric Xing. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2309.10818)]
7. **Deduplicating Training Data Makes Language Models Better**    
   Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini. *ACL 2022.* [[pdf](https://arxiv.org/abs/2107.06499)]
8. **Noise-Robust De-Duplication at Scale**  
   Emily Silcock, Luca D'Amico-Wong, Jinglin Yang, Melissa Dell. *arXiv 2022.* [[pdf](https://arxiv.org/abs/2210.04261)]
9. **In Defense of Minhash over Simhash**  
   Anshumali Shrivastava, Ping Li. *AISTATS 2014.* [[pdf](https://proceedings.mlr.press/v33/shrivastava14.html)]
10. **Similarity estimation techniques from rounding algorithms**  
    Moses S. Charikar. *STOC 2002.* [[pdf](https://doi.org/10.1145/509907.509965)]
11. **On the Resemblance and Containment of Documents**  
    A. Broder. *Compression and Complexity of SEQUENCES 1997.* [[pdf](https://doi.org/10.1109/SEQUEN.1997.666900)]

#### Approximate Frequency-based Down-Weighting

1. SoftDedup: **SoftDedup: an Efficient Data Reweighting Method for Speeding Up Language Model Pre-training**  
   Nan He, Weichen Xiong, Hanwen Liu, Yi Liao, Lei Ding, Kai Zhang, Guohua Tang, Xiao Han, Yang Wei. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.220/)]

#### Embedding-Based Clustering

1. FairDeDup: **FairDeDup: Detecting and Mitigating Vision-Language Fairness Disparities in Semantic Dataset Deduplication**  
   Eric Slyman, Stefan Lee, Scott Cohen, Kushal Kafle. *CVPR 2024.* [[pdf](https://arxiv.org/abs/2404.16123)]
2. **Effective Pruning of Web-Scale Datasets Based on Complexity of Concept Clusters**  
   Amro Abbas, Evgenia Rusak, Kushal Tirumala, Wieland Brendel, Kamalika Chaudhuri, Ari S. Morcos. *ICLR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.04578)]
3. D4: **D4: Improving LLM Pretraining via Document De-Duplication and Diversification**  
   Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari Morcos. *NeurIPS 2023.* [[pdf](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a8f8cbd7f7a5fb2c837e578c75e5b615-Abstract-Datasets_and_Benchmarks.html)]
4. SemDeDup: **SemDeDup: Data-efficient learning at web-scale through semantic deduplication**  
   Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, Ari S. Morcos. *ICLR 2023.* [[pdf](https://iclr.cc/virtual/2023/13610)]
5. OPT: **OPT: Open Pre-trained Transformer Language Models**  
   Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, Luke Zettlemoyer. *arXiv 2022.* [[pdf](https://arxiv.org/abs/2205.01068v4)]
6. CLIP: **Learning Transferable Visual Models From Natural Language Supervision**  
   Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. *ICML 2021.* [[pdf](https://proceedings.mlr.press/v139/radford21a)]
7. OpenCLIP: **OpenCLIP**     
   Ilharco, Gabriel, Wortsman, Mitchell, Wightman, Ross, Gordon, Cade, Carlini, Nicholas, Taori, Rohan, Dave, Achal, Shankar, Vaishaal, Namkoong, Hongseok, Miller, John, Hajishirzi, Hannaneh, Farhadi, Ali, Schmidt, Ludwig. *2021*. [[pdf](https://doi.org/10.5281/zenodo.5143773)]
8. LAION-400M: **LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs**  
   Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, Aran Komatsuzaki. *NeurIPS 2021.* [[pdf](https://doi.org/10.48550/arXiv.2111.02114)]

#### Non-Text Data Deduplication

1. DataComp: **DataComp: In search of the next generation of multimodal datasets**  
   Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, Ludwig Schmidt. *NeurIPS 2023*. [[pdf](https://arxiv.org/abs/2304.14108)]
2. SemDeDup: **SemDeDup: Data-efficient learning at web-scale through semantic deduplication**  
   Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, Ari S. Morcos. *ICLR 2023.* [[pdf](https://iclr.cc/virtual/2023/13610)]
3. CLIP: **Learning Transferable Visual Models From Natural Language Supervision**  
   Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. *ICML 2021.* [[pdf](https://proceedings.mlr.press/v139/radford21a)]
4. CNN-based near-duplicate detector: **Contrastive Learning with Large Memory Bank and Negative Embedding Subtraction for Accurate Copy Detection**  
   Shuhei Yokoo. *arXiv 2021*. [[pdf](https://arxiv.org/abs/2112.04323)]



### 1.3 Data Filtering

[⬆️top](#table-of-contents)

#### Sample-level Filtering

##### (1) Statistical Evaluation

1. **Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models**  
   Zachary Ankner, Cody Blakeney, Kartik Sreenivasan, Max Marion, Matthew L. Leavitt, Mansheej Paul. *ICLR 2025.* [[pdf](https://iclr.cc/virtual/2025/poster/31214)]
2. DEALRec: **Data-efficient Fine-tuning for LLM-based Recommendation**  
   Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, Tat-Seng Chua. *SIGIR 2024.* [[pdf](https://arxiv.org/abs/2401.17197)]
3. SHED: **SHED: Shapley-Based Automated Dataset Refinement for Instruction Fine-Tuning**  
   Yexiao He, Ziyao Wang, Zheyu Shen, Guoheng Sun, Yucong Dai, Yongkai Wu, Hongyi Wang, Ang Li. *NeurIPS 2024.* [[pdf](https://arxiv.org/abs/2405.00705)]
4. SmallToLarge: **SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models**  
   Yu Yang, Siddhartha Mishra, Jeffrey Chiang, Baharan Mirzasoleiman. *NeurIPS 2024.* [[pdf](https://neurips.cc/virtual/2024/poster/95679)]
5. Density-Based Pruning (DBP): **Effective Pruning of Web-Scale Datasets Based on Complexity of Concept Clusters**  
   Amro Abbas, Evgenia Rusak, Kushal Tirumala, Wieland Brendel, Kamalika Chaudhuri, Ari S. Morcos. *ICLR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.04578)]
6. WizardLM: **WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions**  
   Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, Daxin Jiang. *ICLR 2024.* [[pdf](https://iclr.cc/virtual/2024/poster/19164)]
7. Superfiltering: **Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning**  
   Ming Li, Yong Zhang, Shwai He, Zhitao Li, Hongyu Zhao, Jianzong Wang, Ning Cheng, Tianyi Zhou. *ACL 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.00530)]
8. **Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models**  
   Dheeraj Mekala, Alex Nguyen, Jingbo Shang. *ACL 2024*. [[pdf](https://aclanthology.org/2024.findings-acl.623/)]
9. Dolma: **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research**  
   Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strub. *ACL 2024*. [[pdf](https://arxiv.org/abs/2402.00159)]
10. **From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning**  
      Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, Jing Xiao. *NAACL 2024*. [[pdf](https://arxiv.org/abs/2308.12032)]
11. **Improving Pretraining Data Using Perplexity Correlations**  
    Tristan Thrush, Christopher Potts, Tatsunori Hashimoto. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2409.05816)]
12. MosaicML: **Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs**  
    The Mosaic Research Team. *2023*. [[pdf](https://www.databricks.com/blog/mpt-7b)]
13. **Instruction Tuning with GPT-4**  
    Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, Jianfeng Gao. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2304.03277)]
14. DINOV2-L/14: **DINOv2: Learning Robust Visual Features without Supervision**  
    Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2304.07193)]
15. The Pile: **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**  
    Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy. *arXiv 2021*. [[pdf](https://arxiv.org/abs/2101.00027)]
16. **Language Models are Unsupervised Multitask Learners**  
    Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever. *OpenAI blog 2019*. [[pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)]
17. **Bag of Tricks for Efficient Text Classification**  
    Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov. *EACL 2017.* [[pdf](https://aclanthology.org/E17-2068.pdf)]
18. **The Shapley Value: Essays in Honor of Lloyd S. Shapley**  
    A. E. Roth, Ed. *Cambridge: Cambridge University Press, 1988*. [[source](https://www.cambridge.org/core/books/shapley-value/D3829B63B5C3108EFB62C4009E2B966E)]

##### (2) Model Scoring

1. Safety-enhanced Aligned LLM Fine-tuning (SEAL): **SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection**  
   Han Shen, Pin-Yu Chen, Payel Das, Tianyi Chen. *ICLR 2025.* [[pdf](https://iclr.cc/virtual/2025/poster/29422)]
2. QuRating: **QuRating: Selecting High-Quality Data for Training Language Models**  
   Alexander Wettig, Aatmik Gupta, Saumya Malik, Danqi Chen. *ICML 2024.* [[pdf](https://arxiv.org/abs/2402.09739)]
3. Data-Efficient Instruction Tuning for Alignment (DEITA): **What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning**  
   Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He. *ICLR 2024.* [[pdf](https://arxiv.org/abs/2312.15685)]
4. Merlinite-7b: **LAB: Large-Scale Alignment for ChatBots**  
   Shivchander Sudalairaj, Abhishek Bhandwaldar, Aldo Pareja, Kai Xu, David D. Cox, Akash Srivastava. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.01081)]
5. **Biases in Large Language Models: Origins, Inventory, and Discussion**  
   Roberto Navigli, Simone Conia, Björn Ross. *ACM JDIQ, 2023.* [[pdf](https://doi.org/10.1145/3597307)]

##### (3) Hybrid Methods

1. **Emergent and predictable memorization in large language models**  
   Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, Hailey Schoelkopf, Quentin Anthony, Shivanshu Purohit, Edward Raff. *NeurIPS 2023*. [[[pdf](https://dl.acm.org/doi/10.5555/3666122.3667341?__cf_chl_tk=sWnInkGSOKRsrS.z3RwRKDT836eoSy1i.k5oxZcfDzA-1748509375-1.0.1.1-lmH0EWkZpuiyEr5uZPEd_C92GFkM6u6BY416q24qBww)]]
2. **When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale**  
   Max Marion, Ahmet Üstün, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, Sara Hooker. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2309.04564)]
3. InstructionMining: **Instruction Mining: Instruction Data Selection for Tuning Large Language Models**  
   Yihan Cao, Yanbin Kang, Chi Wang, Lichao Sun. *arxiv 2023.* [[pdf](https://arxiv.org/abs/2307.06290)]
4. LLaMA2-7B: **Llama 2: Open Foundation and Fine-Tuned Chat Models**  
   Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2307.09288)]
5. MoDS: **MoDS: Model-oriented Data Selection for Instruction Tuning**  
   Qianlong Du, Chengqing Zong, Jiajun Zhang. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2311.15653)]
6. BlendSearch: **Economic Hyperparameter Optimization With Blended Search Strategy**  
   Chi Wang, Qingyun Wu, Silu Huang, Amin Saied. *ICLR 2021.* [[pdf](https://iclr.cc/virtual/2021/poster/3052)]
7. BERT: **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
   Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. *NAACL 2019.* [[pdf](https://aclanthology.org/N19-1423.pdf)]
8. K-Center greedy algorithm: **Active Learning for Convolutional Neural Networks: A Core-Set Approach**  
   Ozan Sener, Silvio Savarese. *ICLR 2018.* [[pdf](https://doi.org/10.48550/arXiv.1708.00489)]

#### Content-level Filtering

1. spaCy: [[source](https://spacy.io/)]
2. CogVideoX: **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer**  
   Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, Jie Tang. *ICLR 2025*. [[pdf](https://arxiv.org/abs/2408.06072)]
3. HunyuanVideo: **HunyuanVideo: A Systematic Framework For Large Video Generative Models**  
   Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, Caesar Zhong (refer to the report for detailed contributions). *arXiv 2025*. [[pdf](https://arxiv.org/abs/2412.03603v6)]
4. Wan: **Wan: Open and Advanced Large-Scale Video Generative Models**  
   Team Wan et al. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2503.20314)]
5. Video-LLaMA: **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding **  
   Hang Zhang, Xin Li, Lidong Bing. *EMNLP 2023 (System Demonstrations)*. [[pdf](https://arxiv.org/abs/2306.02858)]
6. **Analyzing Leakage of Personally Identifiable Information in Language Models**  
   Nils Lukas, Ahmed Salem, Robert Sim, Shruti Tople, Lukas Wutschitz, Santiago Zanella-Béguelin. *IEEE S&P 2023.* [[pdf](https://arxiv.org/abs/2302.00539)]
7. DeID-GPT: **DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4**  
   Zhengliang Liu, Yue Huang, Xiaowei Yu, Lu Zhang, Zihao Wu, Chao Cao, Haixing Dai, Lin Zhao, Yiwei Li, Peng Shu, Fang Zeng, Lichao Sun, Wei Liu, Dinggang Shen, Quanzheng Li, Tianming Liu, Dajiang Zhu, Xiang Li. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2303.11032)]
8. Baichuan 2: **Baichuan 2: Open Large-scale Language Models**  
   Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, et al. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2309.10305)]
9. Dover: **Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives **  
   Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, Weisi Lin. *arXiv 2022*. [[pdf](https://arxiv.org/abs/2211.04894)]
10. YOLOX: **YOLOX: Exceeding YOLO Series in 2021**  
      Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun. *arXiv 2021*. [[pdf](https://arxiv.org/abs/2107.08430)]
11. LAION-5B: **LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs**  
    Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, Aran Komatsuzaki. *Short version. Accepted at Data-Centric AI Workshop, NeurIPS 2021*. [[pdf](https://doi.org/10.48550/arXiv.2111.02114)]
12. FLAIR: **FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP**  
    Alan Akbik, Tanja Bergmann, Duncan Blythe, Kashif Rasul, Stefan Schweter, Roland Vollgraf. *NAACL 2019 Demos.* [[pdf](https://aclanthology.org/N19-4010/)]

### 1.4 Data Selection

[⬆️top](#table-of-contents)

1. **A Survey on Data Selection for Language Models**  
   Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, Colin Raffel, Shiyu Chang, Tatsunori Hashimoto, William Yang Wang. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2402.16827)]

2. **A Survey on Data Selection for LLM Instruction Tuning**  
   Jiahao Wang, Bolin Zhang, Qianlong Du, Jiajun Zhang, Dianhui Chu. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2402.05123)]

#### Similarity-based Data Selection

1. spaCy: [[source](https://spacy.io/)]
2. Domain Specific Score(DSS): **Enabling On-Device Large Language Model Personalization with Self-Supervised Data Selection and Synthesis**  
   Ruiyang Qin, Jun Xia, Zhenge Jia, Meng Jiang, Ahmed Abbasi, Peipei Zhou, Jingtong Hu, Yiyu Shi. *DAC 2024.* [[pdf](https://doi.org/10.1145/3649329.3655665)]
3. CoLoR-filter: **CoLoR-Filter: Conditional Loss Reduction Filtering for Targeted Language Model Pre-training**  
   David Brandfonbrener, Hanlin Zhang, Andreas Kirsch, Jonathan Richard Schwarz, Sham Kakade. *NeurIPS 2024.* [[pdf](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b0f25f0a63cc544d506e4c1374a3c807-Abstract-Conference.html)]
4. Domain-Adaptive Continual Pre-training (DACP): **Efficient Continual Pre-training for Building Domain Specific Large Language Models**  
   Yong Xie, Karan Aggarwal, Aitzaz Ahmad. *Findings of ACL 2024*. [[pdf](https://aclanthology.org/2024.findings-acl.606/)]
5. DSIR: **Data Selection for Language Models via Importance Resampling**  
   Sang Michael Xie, Shibani Santurkar, Tengyu Ma, Percy Liang. *NeurIPS 2023.* [[pdf](https://doi.org/10.48550/arXiv.2302.03169)]

#### Optimization-based Data Selection

1. Model-Aware Dataset Selection with Datamodels (DsDm): **DSDM: model-aware dataset selection with datamodels**  
   Logan Engstrom, Axel Feldmann, Aleksander Mądry. *ICML 2024.* [[pdf](https://dl.acm.org/doi/10.5555/3692070.3692568)]
2. Low-rank Gradient Similarity Search (LESS): **LESS: Selecting Influential Data for Targeted Instruction Tuning**  
   Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen. *ICML 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.04333)]
3. Task-Specific Data Selection (TSDS): **TSDS: Data Selection for Task-Specific Model Finetuning**  
   Zifan Liu, Amin Karbasi, Theodoros Rekatsinas. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2410.11303)]
4. Datamodels: **Datamodels: Understanding Predictions with Data and Data with Predictions**  
   Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry. *ICML 2022.* [[pdf](https://proceedings.mlr.press/v162/ilyas22a.html)]

#### Model-based Data Selection

1. Autonomous Data Selection (AutoDS): **Autonomous Data Selection with Language Models for Mathematical Texts**  
   Yifan Zhang, Yifan Luo, Yang Yuan, Andrew Chi-Chih Yao. *ICLR 2024.* [[pdf](https://iclr.cc/virtual/2024/22423)]



### 1.5 Data Mixing

[⬆️top](#table-of-contents)

1. **Scalable Data Ablation Approximations for Language Models through Modular Training and Merging**  
   Clara Na, Ian Magnusson, Ananya Harsh Jha, Tom Sherborne, Emma Strubell, Jesse Dodge, Pradeep Dasigi. *EMNLP 2024.* [[pdf](https://arxiv.org/abs/2410.15661v1)]
2. **Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models**  
   Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, Hao Wang. *COLING 2024.* [ [pdf](https://arxiv.org/abs/2403.03432v1) ]

#### Heuristic Optimization

1. Bimix: **BiMix: Bivariate Data Mixing Law for Language Model Pretraining**  
   Ce Ge, Zhijian Ma, Daoyuan Chen, Yaliang Li, Bolin Ding. *arXiv 2024.* [ [pdf](https://arxiv.org/abs/2405.14908) ]
2. Maximize Your Data's Potential: **Maximize Your Data's Potential: Enhancing LLM Accuracy with Two-Phase Pretraining**  
   Steven Feng, Shrimai Prabhumoye, Kezhi Kong, Dan Su, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2412.15285)]
3. Slimpajama: **SlimPajama-DC: Understanding Data Combinations for LLM Training**  
   Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Zhengzhong Liu, Hongyi Wang, Bowen Tan, Joel Hestness, Natalia Vassilieva, Daria Soboleva, Eric Xing. *arXiv 2023.* [ [pdf](https://arxiv.org/abs/2309.10818) ]
4. HumanEval:  **Evaluating Large Language Models Trained on Code**  
   Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba. *arXiv 2021*. [[pdf](https://arxiv.org/abs/2107.03374)]
5. C4: **Exploring the limits of transfer learning with a unified text-to-text transformer**  
   Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. *JMLR 2020.* [[pdf](https://arxiv.org/abs/1910.10683v4)]
6. CommonsenseQA: **CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge**  
   Alon Talmor, Jonathan Herzig, Nicholas Lourie, Jonathan Berant. *NAACL 2019*. [[pdf](https://arxiv.org/abs/1811.00937)]
7. Shannon entropy, conditional entropy: **A mathematical theory of communication**  
   C. E. Shannon. *The Bell system technical journal 1948*. [[pdf](https://ieeexplore.ieee.org/document/6773024)]

#### Bilevel Optimization

1. ScaleBiO: **ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting**  
   Rui Pan, Jipeng Zhang, Xingyuan Pan, Renjie Pi, Xiaoyu Wang, Tong Zhang. *ACL 2025.* [ [pdf](https://arxiv.org/abs/2406.19976) ]
2. DoGE: **DoGE: Domain Reweighting with Generalization Estimation**  
   Simin Fan, Matteo Pagliardini, Martin Jaggi. *ICML 2024.* [[pdf](https://icml.cc/virtual/2024/poster/34869)]
3. Bilevel Optimization: **An overview of bilevel optimization**  
   Benoît Colson, Patrice Marcotte, Gilles Savard. *AOR 2007.* [pdf](https://link.springer.com/article/10.1007/s10479-007-0176-2)]

#### Distributionally Robust Optimization

1. tDRO: **Task-level Distributionally Robust Optimization for Large Language Model-based Dense Retrieval**  
   Guangyuan Ma, Yongliang Ma, Xing Wu, Zhenpeng Su, Ming Zhou, Songlin Hu. *AAAI 2025.* [[pdf](https://arxiv.org/abs/2408.10613)]
2. DoReMi: **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining**  
   Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu. *NeurIPS 2023.* [pdf](https://arxiv.org/abs/2305.10429)]
3. Qwen1.5-0.5B: **Qwen Technical Report**  
   Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, et al. *arXiv 2023.* [pdf](https://arxiv.org/abs/2309.16609v1)]

#### Model-Based Optimization

1. REGMIX: **RegMix: Data Mixture as Regression for Language Model Pre-training**  
   Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, Min Lin. *ICLR 2025.* [[pdf](https://iclr.cc/virtual/2025/poster/30960)]
2. Data Mixing Laws: **Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance**  
   Jiasheng Ye, Peiju Liu, Tianxiang Sun, Yunhua Zhou, Jun Zhan, Xipeng Qiu. *ICLR 2025.* [[pdf](https://arxiv.org/abs/2403.16952)]
3. CMR: **CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training of Language Models**  
   Jiawei Gu, Zacc Yang, Chuanghao Ding, Rui Zhao, Fei Tan. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.emnlp-main.903)]
4. TinyLlama: **TinyLlama: An Open-Source Small Language Model**  
   Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2401.02385)]
5. BiMix: **BiMix: Bivariate Data Mixing Law for Language Model Pretraining**  
   Ce Ge, Zhijian Ma, Daoyuan Chen, Yaliang Li, Bolin Ding. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2405.14908)]
6. D-CPT: **D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models**  
   Haoran Que, Jiaheng Liu, Ge Zhang, Chenchen Zhang, Xingwei Qu, Yinghao Ma, Feiyu Duan, Zhiqi Bai, Jiakai Wang, Yuanxing Zhang, Xu Tan, Jie Fu, Wenbo Su, Jiamang Wang, Lin Qu, Bo Zheng. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2406.01375)]
7. **Data Proportion Detection for Optimized Data Management for Large Language Models**  
   Hao Liang, Keshi Zhao, Yajie Yang, Bin Cui, Guosheng Dong, Zenan Zhou, Wentao Zhang. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2409.17527)]
8. DoReMi: **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining**  
   Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu. *NeurIPS 2023.* [[pdf](https://arxiv.org/abs/2305.10429)]
9. Chinchilla Scaling Law: **Training compute-optimal large language models**  
   Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. *NeurIPS 2022.* [[pdf](https://dl.acm.org/doi/10.5555/3600270.3602446)]
10. LightGBM: **LightGBM: a highly efficient gradient boosting decision tree**  
    Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. *NeurIPS 2017.* [[pdf](https://dl.acm.org/doi/10.5555/3294996.3295074)]



### 1.6 Data Distillation and Synthesis

[⬆️top](#table-of-contents)

1. **How to Synthesize Text Data without Model Collapse?**  
   Xuekai Zhu, Daixuan Cheng, Hengli Li, Kaiyan Zhang, Ermo Hua, Xingtai Lv, Ning Ding, Zhouhan Lin, Zilong Zheng, Bowen Zhou. *ICML 2025*. [[pdf](https://arxiv.org/abs/2412.14689)]
2. **Differentially Private Synthetic Data via Foundation Model APIs 2: Text**  
   Chulin Xie, Zinan Lin, Arturs Backurs, Sivakanth Gopi, Da Yu, Huseyin A Inan, Harsha Nori, Haotian Jiang, Huishuai Zhang, Yin Tat Lee, Bo Li, Sergey Yekhanin. *ICML 2024.* [[pdf](https://arxiv.org/abs/2403.01749v2)]
3. **LLM See, LLM Do: Leveraging Active Inheritance to Target Non-Differentiable Objectives**  
   Luísa Shimabucoro, Sebastian Ruder, Julia Kreutzer, Marzieh Fadaee, Sara Hooker. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.emnlp-main.521)]
4. **WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions**  
   Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, Daxin Jiang. *ICLR 2024.* [[pdf](https://iclr.cc/virtual/2024/poster/19164)]
5. **Augmenting Math Word Problems via Iterative Question Composing**  
   Haoxiong Liu, Yifan Zhang, Yifan Luo, Andrew Chi-Chih Yao. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.09003)]

#### Knowledge Distillation

1. MCKD: **Multistage Collaborative Knowledge Distillation from a Large Language Model for Semi-Supervised Sequence Generation**   
   Jiachen Zhao, Wenlong Zhao, Andrew Drozdov, Benjamin Rozonoyer, Md Arafat Sultan, Jay-Yoon Lee, Mohit Iyyer, Andrew McCallum. *ACL 2024*. [[pdf](https://arxiv.org/abs/2311.08640)]
2. Program-aided Distillation (PaD): **PaD: Program-aided Distillation Can Teach Small Models Reasoning Better than Chain-of-thought Fine-tuning**  
   Xuekai Zhu, Biqing Qi, Kaiyan Zhang, Xinwei Long, Zhouhan Lin, Bowen Zhou. *NAACL 2024*. [[pdf](https://arxiv.org/abs/2305.13888)]
3. **Knowledge Distillation Using Frontier Open-source LLMs: Generalizability and the Role of Synthetic Data**   
   Anup Shirgaonkar, Nikhil Pandey, Nazmiye Ceren Abay, Tolga Aktas, Vijay Aski. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2410.18588)]
4. GSM8K: **Training Verifiers to Solve Math Word Problems**  
   Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman. *arXiv 2021*. [[pdf](https://arxiv.org/abs/2110.14168)]

#### Pre-training Data Augmentation

1. BERT-Tiny-Chinese: [[source](https://huggingface.co/ckiplab/bert-tiny-chinese)]
2. Case2Code: **Case2Code: Scalable Synthetic Data for Code Generation**   
   Yunfan Shao, Linyang Li, Yichuan Ma, Peiji Li, Demin Song, Qinyuan Cheng, Shimin Li, Xiaonan Li, Pengyu Wang, Qipeng Guo, Hang Yan, Xipeng Qiu, Xuanjing Huang, Dahua Lin. *COLING 2025*. [[pdf](https://aclanthology.org/2025.coling-main.733/)]
3. **Advancing Mathematical Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages**  
   Zui Chen, Tianqiao Liu, Mi Tian, Qing Tong, Weiqi Luo, Zitao Liu. *ICLR 2025*. [[pdf](https://arxiv.org/abs/2501.14002)]
4. JiuZhang3.0: **JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models**  
   Kun Zhou, Beichen Zhang, Jiapeng Wang, Zhipeng Chen, Wayne Xin Zhao, Jing Sha, Zhichao Sheng, Shijin Wang, Ji-Rong Wen. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2405.14365)]
5. Florence-large: **Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks**  
   Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, Lu Yuan. *CVPR 2024*. [[pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_CVPR_2024_paper.pdf)]
6. DiffuseMix: **DiffuseMix: Label-Preserving Data Augmentation with Diffusion Models**  
   Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood, Karthik Nandakumar. *CVPR 2024*. [[pdf](https://arxiv.org/abs/2405.14881)]
7. Magicoder-S-DS-6.7B: **Magicoder: Empowering Code Generation with OSS-Instruct **  
   Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang. *ICML 2024*. [[pdf](https://arxiv.org/abs/2312.02120)]
8. Instruction PT: **Instruction Pre-Training: Language Models are Supervised Multitask Learners**  
   Daixuan Cheng, Yuxian Gu, Shaohan Huang, Junyu Bi, Minlie Huang, Furu Wei. *EMNLP 2024*. [[pdf](https://arxiv.org/abs/2406.14491)]
9. Dolma’s CC: **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research**  
   Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strub. *ACL 2024*. [[pdf](https://arxiv.org/abs/2402.00159)]
10. WRAP: **Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling**   
    Pratyush Maini, Skyler Seto, Richard Bai, David Grangier, Yizhe Zhang, Navdeep Jaitly. *ACL 2024*. [[pdf](https://aclanthology.org/2024.acl-long.757/)]
11. VeCLIP: **VeCLIP: Improving CLIP Training via Visual-Enriched Captions**  
    Zhengfeng Lai, Haotian Zhang, Bowen Zhang, Wentao Wu, Haoping Bai, Aleksei Timofeev, Xianzhi Du, Zhe Gan, Jiulong Shan, Chen-Nee Chuah, Yinfei Yang, Meng Cao. *ECCV 2024*. [[pdf](https://dl.acm.org/doi/10.1007/978-3-031-72946-1_7)]
12. Diffusion Models: **Diffusion Models and Representation Learning: A Survey **  
    Michael Fuest, Pingchuan Ma, Ming Gui, Johannes Schusterbauer, Vincent Tao Hu, Bjorn Ommer. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2407.00783)]
13. CtrlSynth: **CtrlSynth: Controllable Image Text Synthesis for Data-Efficient Multimodal Learning**  
    Qingqing Cao, Mahyar Najibi, Sachin Mehta. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2410.11963)]
14. Qwen2-Math-72B, Qwen2-7B-Instruct: **Qwen2 Technical Report**  
    An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, Zhihao Fan. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2407.10671)]
15. TinyLlama-1.1B: **TinyLlama: An Open-Source Small Language Model**  
    Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2401.02385)]
16. **On the Diversity of Synthetic Data and its Impact on Training Large Language Models**  
    Hao Chen, Abdul Waheed, Xiang Li, Yidong Wang, Jindong Wang, Bhiksha Raj, Marah I. Abdin. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2410.15226)]
17. **Towards Effective and Efficient Continual Pre-training of Large Language Models**  
    Jie Chen, Zhipeng Chen, Jiapeng Wang, Kun Zhou, Yutao Zhu, Jinhao Jiang, Yingqian Min, Wayne Xin Zhao, Zhicheng Dou, Jiaxin Mao, Yankai Lin, Ruihua Song, Jun Xu, Xu Chen, Rui Yan, Zhewei Wei, Di Hu, Wenbing Huang, Ji-Rong Wen. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2407.18743)]
18. LaCLIP: **Improving CLIP Training with Language Rewrites**  
    Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, Yonglong Tian. *NeurIPS 2023*. [[pdf](https://arxiv.org/abs/2305.20088)]
19. EDA: **Effective Data Augmentation With Diffusion Models**  
    Brandon Trabucco, Kyle Doherty, Max Gurinas, Ruslan Salakhutdinov. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2302.07944)]
20. Mistral-7B: **Mistral 7B**  
    Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2310.06825)]
21. Llama2: **Llama 2: Open Foundation and Fine-Tuned Chat Models**  
    Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2307.09288)]
22. stable-diffusion-x1-base-1.0: **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**  
    Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2307.01952)]
23. C4: **Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus**  
    Jesse Dodge, Maarten Sap, Ana Marasović, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, Matt Gardner. *EMNLP 2021*. [[pdf](https://arxiv.org/abs/2104.08758)]
24. Pile benchmark: **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**  
    Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy. *arXiv 2021*. [[pdf](https://arxiv.org/abs/2101.00027)]
25. ARC-Challenge: **First Steps of an Approach to the ARC Challenge based on Descriptive Grid Models and the Minimum Description Length Principle**  
    Sébastien Ferré (Univ Rennes, CNRS, IRISA). *arXiv 2021*. [[pdf](https://arxiv.org/abs/2112.00848)]
26. TinyBERT: **TinyBERT: Distilling BERT for Natural Language Understanding**  
    Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. *Findings of EMNLP 2020*. [[pdf](https://arxiv.org/abs/1909.10351)]
27. HellaSwag: **HellaSwag: Can a Machine Really Finish Your Sentence?**  
    Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, Yejin Choi. *ACL 2019*. [[pdf](https://arxiv.org/abs/1905.07830)]

#### SFT Data Augmentation

1. KPDDS: **Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning**  
   Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, Weizhu Chen. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.02333)]
2. MMIQC: **Augmenting Math Word Problems via Iterative Question Composing**  
   Haoxiong Liu, Yifan Zhang, Yifan Luo, Andrew Chi-Chih Yao. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.09003)]
3. AgentInstruct: **AgentInstruct: Toward Generative Teaching with Agentic Flows**  
   Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andres Codas, Yadong Lu, Wei-ge Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, Ahmed Awadallah. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2407.03502)]
4. GLAN: **Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models**  
   Haoran Li, Qingxiu Dong, Zhengyang Tang, et al. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.13064)]
5. SELF-INSTRUCT: **Self-Instruct: Aligning Language Models with Self-Generated Instructions**  
   Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. *ACL 2023.* [[pdf](https://aclanthology.org/2023.acl-long.754)]

#### SFT Reasoning Data Augmentation

1. DeepSeek-R1: **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**  
   DeepSeek-AI. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2501.12948)]
2. LIMO: **LIMO: Less is More for Reasoning**  
   Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, Pengfei Liu. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.03387)]
3. **LLMs Can Easily Learn to Reason from Demonstrations: Structure, Not Content, Is What Matters!**  
   Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Eric Tang, Sumanth Hegde, Kourosh Hakhamaneshi, Shishir G. Patil, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.07374)]
4. Satori, Chain-of-ActionThought (COAT): **Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search**  
   Maohao Shen, Guangtao Zeng, Zhenting Qi, Zhang-Wei Hong, Zhenfang Chen, Wei Lu, Gregory Wornell, Subhro Das, David Cox, Chuang Gan. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.02508)]
5. **Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling**  
   Zhenyu Hou, Xin Lv, Rui Lu, Jiajie Zhang, Yujiang Li, Zijun Yao, Juanzi Li, Jie Tang, Yuxiao Dong. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.11651)]
6. MUSTARD: **MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data**  
   Yinya Huang, Xiaohan Lin, Zhengying Liu, Qingxing Cao, Huajian Xin, Haiming Wang, Zhenguo Li, Linqi Song, Xiaodan Liang. *ICLR 2024.* [[pdf](https://arxiv.org/abs/2402.08957v3)]
7. Math-Shepherd: **Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations**  
   Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, Zhifang Sui. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.510)]
8. Numina-Math: **NuminaMath: The largest public dataset in AI4Maths with 860k pairs of competition math problems and solutions**   
   Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q. Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, Stanislas Polu. *2024*. [[pdf](http://faculty.bicmr.pku.edu.cn/~dongbin/Publications/numina_dataset.pdf)] [[source](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)]
9. QwQ-32B-Preview: **QwQ: Reflect Deeply on the Boundaries of the Unknown**   
   Qwen Team. *2024*. [[blog](https://qwenlm.github.io/blog/qwq-32b-preview/)] [[source](https://huggingface.co/Qwen/QwQ-32B-Preview)]
10. **Let's Verify Step by Step**  
    Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2305.20050)]

#### Reinforcement Learning

1. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**  
   Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. NeurIPS 2023. [[pdf](https://dl.acm.org/doi/10.5555/3666122.3668142)]
2. **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**  
   Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, et al. *arXiv 2022.* [[pdf](https://doi.org/10.48550/arXiv.2204.05862)]

#### Retrieval-Augmentation Generation

1. **Mitigating the Privacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data**  
   Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren, Tianqi Zheng, Hanqing Lu, Han Xu, Hui Liu, Yue Xing, Jiliang Tang. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2406.14773)]



### 1.7 End-to-End Data Processing Pipelines

[⬆️top](#table-of-contents)

#### 1.7.1 Typical data processing frameworks

1. Data-Juicer: **Data-Juicer: A One-Stop Data Processing System for Large Language Models**  
   Daoyuan Chen, Yilun Huang, Zhijian Ma, Hesen Chen, Xuchen Pan, Ce Ge, Dawei Gao, Yuexiang Xie, Zhaoyang Liu, Jinyang Gao, Yaliang Li, Bolin Ding, Jingren Zhou. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3626246.3653385)]
2. **An Integrated Data Processing Framework for Pretraining Foundation Models**  
   Yiding Sun, Feng Wang, Yutao Zhu, Wayne Xin Zhao, Jiaxin Mao. *SIGIR 2024.* [[pdf](https://doi.org/10.1145/3626772.3657671)]
3. Dataverse: **Dataverse: Open-Source ETL (Extract, Transform, Load) Pipeline for Large Language Models**  
   Hyunbyung Park, Sukyung Lee, Gyoungjin Gim, Yungi Kim, Dahyun Kim, Chanjun Park. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2403.19340v1)]

#### 1.7.2 Typical data pipelines

1. Common Crawl: [[source](https://commoncrawl.org/)]
2. Falcon LLMs, heuristic filtering: **The RefinedWeb dataset for falcon LLM: outperforming curated corpora with web data only**  
   Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, Julien Launay. *NeurIPS 2023.* [[pdf](https://dl.acm.org/doi/10.5555/3666122.3669586)]
3. Trafilatura: **Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction**  
   Adrien Barbaresi. *ACL 2021.* [[pdf](https://aclanthology.org/2021.acl-demo.15.pdf)]
4. document-level filtering, gram level: **Scaling Language Models: Methods, Analysis & Insights from Training Gopher**  
   Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, et al. *arXiv 2021.* [[pdf](https://arxiv.org/abs/2112.11446v2)]
5. CCNet: **CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data**  
   Guillaume Wenzek, Marie - Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, Edouard Grave. *LREC 2020.* [[pdf](https://aclanthology.org/2020.lrec-1.494/)]
6. C4: **Exploring the limits of transfer learning with a unified text-to-text transformer**  
   Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. *JMLR 2020.* [[pdf](https://arxiv.org/abs/1910.10683v4)]
7. fastText: **Bag of Tricks for Efficient Text Classification**  
   Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov. *EACL 2017.* [[pdf](https://aclanthology.org/E17-2068.pdf)]

#### 1.7.3 Orchestration of data pipelines

1. Data-Juicer Sandbox: **Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development**  
   Daoyuan Chen, Haibin Wang, Yilun Huang, Ce Ge, Yaliang Li, Bolin Ding, Jingren Zhou. *ICML 2025*. [[pdf](https://arxiv.org/abs/2407.11784v2)]



## 2 Data Storage for LLM

[⬆️top](#table-of-contents)

### 2.1 Data Formats

#### Training Data Format

1. TFRecord: [[source](https://www.tensorflow.org/tutorials/load_data/tfrecord)]
2. MindRecord: [[source](https://www.mindspore.cn/)]
3. tf.data.Dataset: [[source](https://www.tensorflow.org/guide/data)]
4. COCO JSON: [[source](https://cocodataset.org/)]

#### Model Data Format

1. PyTorch-specific formats (.pt, .pth): [[source](https://pytorch.org/)]
2. TensorFlow(SavedModel, .ckpt): [[source](https://www.tensorflow.org)]
3. Hugging Face Transformers library: [[source]( https://huggingface.co/)]
4. Pickle (.pkl): [[source](https://docs.python.org/3/library/pickle.html)]
5. ONNX: [[source](https://onnx.ai)]
6. Safetensors: **An Empirical Study of Safetensors' Usage Trends and Developers' Perceptions**  
   Beatrice Casey, Kaia Damian, Andrew Cotaj, Joanna C. S. Santos. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.02170)]



### 2.2 Data Distribution

[⬆️top](#table-of-contents)

1. DeepSeek-R1: **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**  
   DeepSeek-AI. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2501.12948)]
2. CC-GPX: **CC-GPX: Extracting High-Quality Annotated Geospatial Data from Common Crawl**  
   Ilya Ilyankou, Meihui Wang, Stefano Cavazzi, James Haworth. *SIGSPATIAL 2024.* [[pdf](https://doi.org/10.1145/3678717.3691215)]

#### Distributed Storage Systems

1. JuiceFS: [[Github](https://github.com/juicedata/juicefs)]
2. 3FS: [[Github](https://github.com/deepseek-ai/3fs)]
3. S3: [[source](https://aws.amazon.com/s3)]
4. HDFS: **Hdfs architecture guide. Hadoop apache project**  
    D. Borthakur et al. *Hadoop apache project, 53(1-13):2, 2008*.[[source](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html)]

#### Heterogeneous Storage Systems

1. **ProTrain: Efficient LLM Training via Memory-Aware Techniques**  
   Hanmei Yang, Jin Zhou, Yao Fu, Xiaoqun Wang, Ramine Roane, Hui Guan, Tongping Liu. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2406.08334)]
2. **ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning**  
   Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He. *SC 2021.* [[pdf](https://doi.org/10.1145/3458817.3476205)]
3. **ZeRO-Offload: Democratizing Billion-Scale Model Training**  
   Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, Yuxiong He. *USENIX ATC 2021.* [[pdf](https://www.usenix.org/system/files/atc21-ren-jie.pdf)]
4. Zero Redundancy Optimizer (ZeRO): **ZeRO: memory optimizations toward training trillion parameter models**  
   Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. *SC 2020.* [[pdf](https://dl.acm.org/doi/10.5555/3433701.3433727)]
5. **vDNN: virtualized deep neural networks for scalable, memory-efficient neural network design**  
   Minsoo Rhu, Natalia Gimelshein, Jason Clemons, Arslan Zulfiqar, Stephen W. Keckler. *MICRO-49 2016.* [[pdf](https://dl.acm.org/doi/10.5555/3195638.3195660)]



### 2.3 Data Organization

[⬆️top](#table-of-contents)

1. hallucination: **Survey of Hallucination in Natural Language Generation**  
   Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Delong Chen, Wenliang Dai, Ho Shu Chan, Andrea Madotto, Pascale Fung. *ACM Computing Surveys (2022)*. [[pdf](https://dl.acm.org/doi/10.1145/3571730)]
2. RAG: **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
   Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela. *NeurIPS 2020.* [[pdf](https://doi.org/10.48550/arXiv.2005.11401)]

#### Vector-Based Organization

1. STELLA: [[source](https://huggingface.co/infgrad/stella-large-zh-v2)]
2. Milvus: [[source](https://milvus.io)]
3. Weaviate: [[source](https://weaviate.io)]
4. LanceDB: [[source](https://lancedb.com)]
5. MoG, MoGG: **Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation**  
   Zijie Zhong, Hanwen Liu, Xiaoya Cui, Xiaofan Zhang, Zengchang Qin. *COLING 2025.* [[pdf](https://doi.org/10.48550/arXiv.2406.00456)]
6. Dense x retrieval: **Dense X Retrieval: What Retrieval Granularity Should We Use?**  
   Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, Dong Yu. *EMNLP 2024*. [[pdf](https://aclanthology.org/2024.emnlp-main.845/)]
7. APS: **Scalable and Domain-General Abstractive Proposition Segmentation**  
   Mohammad Javad Hosseini, Yang Gao, Tim Baumgärtner, Alex Fabrikant, Reinald Kim Amplayo. *Findings of EMNLP 2024*. [[pdf](https://aclanthology.org/2024.findings-emnlp.517/)]
8. **A Hierarchical Context Augmentation Method to Improve Retrieval-Augmented LLMs on Scientific Papers**  
   Tian-Yi Che, Xian-Ling Mao, Tian Lan, Heyan Huang. *KDD 2024*. [[pdf](https://dl.acm.org/doi/10.1145/3637528.3671847)]
9. cross-lingual retrieval: **M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**  
   Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu. *Findings of ACL 2024.* [[pdf](https://aclanthology.org/2024.findings-acl.137.pdf)]
10. Thread: **Thread: A Logic-Based Data Organization Paradigm for How-To Question Answering with Retrieval Augmented Generation**  
    Kaikai An, Fangkai Yang, Liqun Li, Junting Lu, Sitao Cheng, Shuzheng Si, Lu Wang, Pu Zhao, Lele Cao, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Qi Zhang, Baobao Chang. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2406.13372)]
11. GleanVec, LeanVec-Sphering: **GleanVec: Accelerating Vector Search with Minimalist Nonlinear Dimensionality Reduction**  
    Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, Ted Willke. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2410.22347)]
12. Faiss: **The Faiss Library**  
    Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, Hervé Jégou. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.08281)]
13. Locally-adaptive Vector Quantization (LVQ): **Similarity Search in the Blink of an Eye with Compressed Indices**  
    Cecilia Aguerrebere, Ishwar Singh Bhati, Mark Hildebrand, Mariano Tepper, Theodore Willke. *VLDB Endowment 2023.* [[pdf](https://doi.org/10.14778/3611479.3611537)]
14. LeanVec: **LeanVec: Searching Vectors Faster by Making Them Fit**  
    Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, Mark Hildebrand, Ted Willke. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2312.16335)]
15. GTE: **Towards General Text Embeddings with Multi-stage Contrastive Learning**  
    Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2308.03281)]

#### Graph-Based Organization

1. ArangoDB: [[source](https://arangodb.com/)]
2. MiniRAG: **MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation**  
   Tianyu Fan, Jingyuan Wang, Xubin Ren, Chao Huang. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.06713)]
3. GraphRAG: **From Local to Global: A Graph RAG Approach to Query-Focused Summarization**  
   Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2404.16130)]
4. LightRAG: **LightRAG: Simple and Fast Retrieval-Augmented Generation**  
   Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2410.05779)]
5. **Graph Databases Assessment: JanusGraph, Neo4j, and TigerGraph**  
   Jéssica Monteiro, Filipe Sá, Jorge Bernardino. *Perspectives and Trends in Education and Technology 2023.* [[pdf](https://doi.org/10.1007/978-981-19-6585-2_58)]
6. RDF (Resource Description Framework) models:**Empirical Evaluation of a Cloud-Based Graph Database: the Case of Neptune**  
   Ghislain Auguste Atemezing. *KGSWC 2021.* [[pdf](https://doi.org/10.1007/978-3-030-91305-2_3)]



### 2.4 Data Movement

[⬆️top](#table-of-contents)

#### Caching Data

1. CacheLib: [[source](https://cachelib.org/)]
2. Tectonic-Shift: **Tectonic-Shift: A Composite Storage Fabric for Large-Scale ML Training**  
   Mark Zhao, Satadru Pan, Niket Agarwal, Zhaoduo Wen, David Xu, Anand Natarajan, Pavan Kumar, Shiva Shankar P, Ritesh Tijoriwala, Karan Asher, Hao Wu, Aarti Basant, Daniel Ford, Delia David, Nezih Yigitbasi, Pratap Singh, Carole-Jean Wu, Christos Kozyrakis. *USENIX ATC 2023.* [[pdf](https://www.usenix.org/conference/atc23/presentation/zhao)]
3. Fluid: **Fluid: Dataset Abstraction and Elastic Acceleration for Cloud-native Deep Learning Training Jobs**  
   Rong Gu, Kai Zhang, Zhihao Xu, Yang Che, Bin Fan, Haojun Hou. *ICDE 2022.* [[pdf](https://doi.org/10.1109/ICDE53745.2022.00209)]
4. Quiver: **Quiver: An Informed Storage Cache for Deep Learning**  
   Abhishek Kumar, Muthian Sivathanu. *USENIX FAST 2020.* [[pdf](https://www.usenix.org/conference/fast20/presentation/kumar)]

#### Data/Operator Offloading

1. Cedar: **cedar: Optimized and Unified Machine Learning Input Data Pipelines**  
   Mark Zhao, Emanuel Adamiak, Christos Kozyrakis. *Proceedings of the VLDB Endowment, Volume 18, Issue 2, 2025.* [[pdf](https://dl.acm.org/doi/10.14778/3705829.3705861)]
2. Pecan: **Pecan: cost-efficient ML data preprocessing with automatic transformation ordering and hybrid placement**  
   Dan Graur, Oto Mraz, Muyu Li, Sepehr Pourghannad, Chandramohan A. Thekkath, Ana Klimovic. *USENIX ATC 2024.* [[pdf](https://dl.acm.org/doi/10.5555/3691992.3692032)]
3. tf.data service: **tf.data service: A Case for Disaggregating ML Input Data Processing**  
   Andrew Audibert, Yang Chen, Dan Graur, Ana Klimovic, Jiří Šimša, Chandramohan A. Thekkath. *SoCC 2023.* [[pdf](https://doi.org/10.1145/3620678.3624666)]
4. Cachew: **Cachew: Machine Learning Input Data Processing as a Service**  
   Dan Graur, Damien Aymon, Dan Kluser, Tanguy Albrici, Chandramohan A. Thekkath, Ana Klimovic. *USENIX ATC 2022.* [[pdf](https://www.usenix.org/conference/atc22/presentation/graur)]
5. Borg: **Borg: the next generation**  
   Muhammad Tirmazi, Adam Barker, Nan Deng, Md E. Haque, Zhijing Gene Qin, Steven Hand, Mor Harchol-Balter, John Wilkes. *EuroSys 2020*. [[pdf](https://dl.acm.org/doi/10.1145/3342195.3387517)]

#### Overlapping of storage and computing

1. RLHFuse: **Optimizing RLHF Training for Large Language Models with Stage Fusion**  
   Yinmin Zhong, Zili Zhang, Bingyang Wu, Shengyu Liu, Yukun Chen, Changyi Wan, Hanpeng Hu, Lei Xia, Ranchen Ming, Yibo Zhu, Xin Jin. *NSDI 2025*. [[pdf](https://www.usenix.org/conference/nsdi25/presentation/zhong)]
2. SiloD: **SiloD: A Co-design of Caching and Scheduling for Deep Learning Clusters**  
   Hanyu Zhao, Zhenhua Han, Zhi Yang, Quanlu Zhang, Mingxia Li, Fan Yang, Qianxi Zhang, Binyang Li, Yuqing Yang, Lili Qiu, Lintao Zhang, Lidong Zhou. *EuroSys 2023.* [[pdf](https://doi.org/10.1145/3552326.3567499)]
3. **Optimization by Simulated Annealing**  
   S. Kirkpatrick, C. D. Gelatt, Jr., M. P. Vecchi. *Science, 220(4598):671–680, 1983*. [[pdf](https://www.science.org/doi/10.1126/science.220.4598.671)]



### 2.5 Data Fault Tolerance

[⬆️top](#table-of-contents)

#### Checkpoints

1. PaddleNLP: [[docs](https://paddlenlp.readthedocs.io)]
2. MegaScale: **MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs**  
   Ziheng Jiang, Haibin Lin, Yinmin Zhong, Qi Huang, et al. *USENIX NSDI 2024.* [[pdf](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng)]
3. ByteCheckpoint: **ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development**  
   Borui Wan, Mingji Han, Yiyao Sheng, Yanghua Peng, Haibin Lin, Mofan Zhang, Zhichao Lai, Menghan Yu, Junda Zhang, Zuquan Song, Xin Liu, Chuan Wu. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2407.20143)]
4. Gemini: **GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints**  
   Zhuang Wang, Zhen Jia, Shuai Zheng, Zhen Zhang, Xinwei Fu, T. S. Eugene Ng, Yida Wang. *SOSP 2023.* [[pdf](https://doi.org/10.1145/3600006.3613145)]
5. CheckFreq: **CheckFreq: Frequent, Fine-Grained DNN Checkpointing**  
   Jayashree Mohan, Amar Phanishayee, Vijay Chidambaram. *USENIX FAST 2021.* [[pdf](https://www.usenix.org/conference/fast21/presentation/mohan)]

#### Redundant Computations

1. ReCycle: **ReCycle: Resilient Training of Large DNNs using Pipeline Adaptation**  
   Swapnil Gandhi, Mark Zhao, Athinagoras Skiadopoulos, Christos Kozyrakis. *SOSP 2024*. [[pdf](https://arxiv.org/abs/2405.14009)]
2. Bamboo: **Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs**  
   John Thorpe, Pengzhan Zhao, Jonathan Eyolfson, and Yifan Qiao;Zhihao Jia, Minjia Zhang, Ravi Netravali, Guoqing Harry Xu.  *NSDI 2023* . [[pdf](https://www.usenix.org/conference/nsdi23/presentation/thorpe)]
3. Oobleck: **Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates**  
   Insu Jang, Zhenning Yang, Zhen Zhang, Xin Jin, Mosharaf Chowdhury. *SOSP 2023*. [[pdf](https://dl.acm.org/doi/10.1145/3600006.3613152)]



### 2.6 KV Cache

[⬆️top](#table-of-contents)

#### Cache Space Management

1. vLLM: **Efficient Memory Management for Large Language Model Serving with PagedAttention**  
   Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, Ion Stoica. *SOSP 2023.* [[pdf](https://arxiv.org/abs/2309.06180)]
2. vTensor: **VTensor: Using Virtual Tensors to Build a Layout-oblivious AI Programming Framework**  
   Feng Yu, Jiacheng Zhao, Huimin Cui, Xiaobing Feng, Jingling Xue. *PACT 2020.* [[pdf](https://dl.acm.org/doi/pdf/10.1145/3410463.3414664)]

#### KV Placement

1. CachedAttention: **Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention**  
   Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu, Pengfei Zuo. *USENIX ATC 2024.* [[pdf](https://arxiv.org/abs/2403.19708)]
2. RAGCache: **RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**  
   Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, Xin Jin. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2404.12457)]

#### KV Shrinking

1. HCache: **Fast State Restoration in LLM Serving with HCache**  
   Shiwei Gao, Youmin Chen, Jiwu Shu. *EuroSys 2025.* [[pdf](https://arxiv.org/abs/2410.05004)]
2. CacheGen: **CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**  
   Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, Junchen Jiang. *SIGCOMM 2024.* [[pdf](https://dl.acm.org/doi/abs/10.1145/3651890.3672274)]
3. MiniCache: **MiniCache: KV Cache Compression in Depth Dimension for Large Language Models**  
   Akide Liu · Jing Liu · Zizheng Pan · Yefei He · Reza Haffari · Bohan Zhuang. *NeurIPS 2024*. [[pdf](https://neurips.cc/virtual/2024/poster/93380)]
4. SLERP: **Animating rotation with quaternion curves**  
   Ken Shoemake. *ACM SIGGRAPH Computer Graphics, Volume 19, Issue 3. 1985*. [[pdf](https://dl.acm.org/doi/10.1145/325165.325242)]

#### KV Indexing

1. ChunkAttention: **ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition**  
   Lu Ye, Ze Tao, Yong Huang, Yang Li. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.623/)]
2. Prefix Sharing Maximization (PSM):**BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching**  
   Zhen Zheng, Xin Ji, Taosong Fang, Fanghao Zhou, Chuanjie Liu, Gang Peng. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2412.03594)]



## 3 Data Serving for LLM

[⬆️top](#table-of-contents)

### 3.1 Data Shuffling

#### Data Shuffling for Training

1. Velocitune: **Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training**  
   Zheheng Luo, Xin Zhang, Xiao Liu, Haoling Li, Yeyun Gong, Chen Qi, Peng Cheng. *ACL 2025*. [[pdf](https://arxiv.org/abs/2411.14318)]
2. **How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition**  
   Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, Jingren Zhou. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.12/)]
3. MOS: **Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models **    
   Minghao Wu, Thuy-Trang Vu, Lizhen Qu, Gholamreza Haffari. *EMNLP 2024.* [[pdf](https://aclanthology.org/2024.emnlp-main.787/)]
4. DMT: **Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning**  
   Jisu Kim, Juhwan Lee. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.07490)]
5. **NLU on Data Diets: Dynamic Data Subset Selection for NLP Classification Tasks**  
   Jean-michel Attendu, Jean-philippe Corbeil. *SustaiNLP @ ACL 2023.* [[pdf](https://aclanthology.org/2023.sustainlp-1.9/)]
6. ODM: **Efficient Online Data Mixing For Language Model Pre-Training**  
   Alon Albalak, Liangming Pan, Colin Raffel, William Yang Wang. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2312.02406)]
7. Moving-one-Sample-out (MoSo): **Data Pruning via Moving-one-Sample-out**  
   Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, Xiaojuan Qi. *NeurIPS 2023*. [[pdf](https://arxiv.org/abs/2310.14664)]
8. EL2N: **BERT on a Data Diet: Finding Important Examples by Gradient-Based Pruning**  
   Mohsen Fayyaz, Ehsan Aghazadeh, Ali Modarressi, Mohammad Taher Pilehvar, Yadollah Yaghoobzadeh, Samira Ebrahimi Kahou. *ENLSP @ NeurIPS2022.* [[pdf](https://doi.org/10.48550/arXiv.2211.05610)]
9. **Scaling Laws for Neural Language Models**  
   Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei. *arXiv 2020*. [[pdf](https://arxiv.org/abs/2001.08361)]
10. **Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory**  
    James L. McClelland, Bruce L. McNaughton, Randall C. O’Reilly. *Psychological Review 1995.* [[pdf](https://cseweb.ucsd.edu/~gary/258/jay.pdf)]
11. **Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem**  
    M. McCloskey, N. J. Cohen. *Psychology of Learning and Motivation 1989.* [[pdf](https://www.sciencedirect.com/science/article/abs/pii/S0079742108605368)]

#### Data Selection for RAG

1. Cohere rerank: [[source](https://docs.cohere.com)]
2. ASRank: **ASRank: Zero-Shot Re-Ranking with Answer Scent for Document Retrieval**  
   Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Adam Jatowt. *NAACL 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.15245)]
3. MAIN-RAG: **MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation**  
   Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh, Menghai Pan, Chin-Chia Michael Yeh, Guanchu Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Mahashweta Das, Na Zou. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2501.00332)]
4. ARAGOG: **ARAGOG: Advanced RAG Output Grading**  
   Matouš Eibich, Shivay Nagpal, Alexander Fred-Ojala. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2404.01037)]
5. **Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!**  
   Yubo Ma, Yixin Cao, YongChing Hong, Aixin Sun. *Findings of EMNLP 2023*. [[pdf](https://aclanthology.org/2023.findings-emnlp.710/)]
6. Chatlaw: **Chatlaw: A Multi-Agent Collaborative Legal Assistant with Knowledge Graph Enhanced Mixture-of-Experts Large Language Model**  
   Jiaxi Cui, Munan Ning, Zongjian Li, Bohua Chen, Yang Yan, Hao Li, Bin Ling, Yonghong Tian, Li Yuan. *arXiv 2023*.[[pdf](https://arxiv.org/abs/2306.16092v2)]
7. RankVicuna: **RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models**  
   Ronak Pradeep, Sahel Sharifymoghaddam, Jimmy Lin. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2309.15088)]



### 3.2 Data Compression

[⬆️top](#table-of-contents)

#### RAG Knowledge Compression

1. COCOM: **Context Embeddings for Efficient Answer Generation in RAG**  
   David Rau, Shuai Wang, Hervé Déjean, Stéphane Clinchant. *WSDM 2025.* [[pdf](https://doi.org/10.48550/arXiv.2407.09252)]
2. xRAG: **xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token**  
   Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, Dongyan Zhao. *NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.13792)]
3. Recomp: **RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation**  
   Fangyuan Xu, Weijia Shi, Eunsol Choi. *ICLR 2024.* [[pdf](https://iclr.cc/virtual/2024/poster/17885)]
4. CompAct: **Compressing Long Context for Enhancing RAG with AMR-based Concept Distillation**   
   Kaize Shi, Xueyao Sun, Qing Li, Guandong Xu. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.03085)]
5. FAVICOMP: **Familiarity-Aware Evidence Compression for Retrieval-Augmented Generation**  
   Dongwon Jung, Qin Liu, Tenghao Huang, Ben Zhou, Muhao Chen. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2409.12468)]

#### Prompt Compression

1. Longllmlingua: **LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression**  
   Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.91/)]
2. Llmlingua-2: **LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression**  
   Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Dongmei Zhang. *Findings of ACL 2024.* [[pdf](https://aclanthology.org/2024.findings-acl.57/)]
3. Llmlingua: **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models**  
   Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu. *EMNLP 2023.* [[pdf](https://aclanthology.org/2023.emnlp-main.825.pdf)]
4. **Learning to Compress Prompts with Gist Tokens**  
   Jesse Mu, Xiang Lisa Li, Noah Goodman. *NeurIPS 2023.* [[pdf](https://arxiv.org/abs/2304.08467)]
5. **Adapting Language Models to Compress Contexts**  
   Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen. *EMNLP 2023.* [[pdf](https://aclanthology.org/2023.emnlp-main.232.pdf)]



### 3.3 Data Packing

[⬆️top](#table-of-contents)

#### Short Sequence Insertion

1. The Best-fit Packing: **Fewer Truncations Improve Language Modeling**  
   Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, Stefano Soatto. *ICML 2024.* [[pdf](https://doi.org/10.48550/arXiv.2404.10830)]
2. **Bucket Pre-training is All You Need**  
   Hongtao Liu, Qiyao Peng, Qing Yang, Kai Liu, Hongyan Xu. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2407.07495)]

#### Sequence Combination Optimization

1. **Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum**  
   Hadi Pouransari, Chun-Liang Li, Jen-Hao Rick Chang, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Oncel Tuzel. *NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.13226)]
2. **Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance**  
   Mario Michael Krell, Matej Kosec, Sergio P. Perez, Andrew Fitzgibbon. *arXiv 2021.* [[pdf](https://doi.org/10.48550/arXiv.2107.02027)]

#### Semantic-Based Packing

1. SPLICE: **Structured Packing in LLM Training Improves Long Context Utilization**  
   Konrad Staniszewski, Szymon Tworkowski, Sebastian Jaszczur, Yu Zhao, Henryk Michalewski, Łukasz Kuciński, Piotr Miłoś. *AAAI 2025.* [[pdf](https://doi.org/10.48550/arXiv.2312.17296)]
2. **In-context Pretraining: Language Modeling Beyond Document Boundaries**  
   Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Gergely Szilvasy, Rich James, Xi Victoria Lin, Noah A. Smith, Luke Zettlemoyer, Scott Yih, Mike Lewis. *ICLR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2310.10638)]



### 3.4 Data Provenance

[⬆️top](#table-of-contents)

1. **A comprehensive survey on data provenance: : State-of-the-art approaches and their deployments for IoT security enforcement**   
   Md Morshed Alam, Weichao Wang. *Journal of Computer Security, Volume 29, Issue 4. 2021*. [[pdf](https://dl.acm.org/doi/abs/10.3233/JCS-200108)]

#### Embedding Markers

1. Bileve: **Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**  
   Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren. *NeurIPS 2024*. [[pdf](https://arxiv.org/abs/2406.01946)]
2. **Undetectable Watermarks for Language Models**  
   Miranda Christ, Sam Gunn, Or Zamir. in *Proceedings of the 37th Annual Conference on Learning Theory (COLT 2024)*. [[pdf](https://arxiv.org/abs/2306.09194)]
3. UPV: **An Unforgeable Publicly Verifiable Watermark for Large Language Models**  
   Aiwei Liu, Leyi Pan, Xuming Hu, Shu'ang Li, Lijie Wen, Irwin King, Philip S. Yu. *ICLR 2024*. [[pdf](https://arxiv.org/abs/2307.16230)]
4. **A Watermark for Large Language Models**  
   John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein. *ICML 2023*. [[pdf](https://arxiv.org/abs/2301.10226)]
5. **Publicly-Detectable Watermarking for Language Models**   
   Jaiden Fairoze, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Mingyuan Wang. *arXiv 2023*. [[pdf](https://arxiv.org/abs/2310.18491)]

#### Statistical Provenance

1. **A Watermark for Large Language Models**  
   John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein. *ICML 2023*. [[pdf](https://arxiv.org/abs/2301.10226)]



## 4 LLM for Data Management

[⬆️top](#table-of-contents)

### 4.1 LLM for Data Manipulation

#### 4.1.1 LLM for Data Cleaning

##### Data Standardization

1. Evaporate: **Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes**  
   Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Hojel, Immanuel Trummer, Christopher Ré. *Proceedings of the VLDB Endowment, Volume 17, Issue 2, 2024.* [[pdf](https://dl.acm.org/doi/abs/10.14778/3626292.3626294)]
2. CleanAgent: **CleanAgent: Automating Data Standardization with LLM-based Agents**  
   Danrui Qi, Jiannan Wang. *arXiv 2024.* [[pdf](https://arxiv.org/pdf/2403.08291)]
3. AutoDCWorkflow: **AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark**  
   Lan Li, Liri Fang, Vetle I. Torvik. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2412.06724)]
4. LLM-GDO: **LLMs with User-defined Prompts as Generic Data Operators for Reliable Data Processing**  
   Luyi Ma, Nikhil Thakurdesai, Jiao Chen, Jianpeng Xu, Evren Körpeoglu, Sushant Kumar, Kannan Achan. *1st IEEE International Workshop on Data Engineering and Modeling for AI (DEMAI), IEEE BigData 2023.* [[pdf](https://arxiv.org/abs/2312.16351)]

##### Data Error Processing

1. GIDCL: **GIDCL: A Graph-Enhanced Interpretable Data Cleaning Framework with Large Language Models**  
   Mengyi Yan, Yaoshu Wang, Yue Wang, Xiaoye Miao, Jianxin Li. *Proceedings of the ACM on Management of Data, Volume 2, Issue 6, 2024.* [[pdf](https://dl.acm.org/doi/10.1145/3698811)]
2. LLMErrorBench: **Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets**  
   Tommaso Bendinelli, Artur Dox, Christian Holz. *ICLR 2025 Workshop on Foundation Models in the Wild*. [[pdf](https://arxiv.org/abs/2503.06664)]
3. Multi-News+: **Multi-News+: Cost-efficient Dataset Cleansing via LLM-based Data Annotation**  
   Juhwan Choi, Jungmin Yun, Kyohoon Jin, YoungBin Kim. *EMNLP 2024*. [[pdf](https://arxiv.org/abs/2404.09682)]
4. Cocoon: **Data Cleaning Using Large Language Models**  
   Shuo Zhang, Zezhou Huang, Eugene Wu. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2410.15547)]
5. LLMClean: **LLMClean: Context-Aware Tabular Data Cleaning via LLM-Generated OFDs**  
   Fabian Biester, Mohamed Abdelaal, Daniel Del Gaudio. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2404.18681)]

##### Data Imputation

1. RetClean: **RetClean: Retrieval-Based Data Cleaning Using LLMs and Data Lakes**  
   Zan Ahmad Naeem, Mohammad Shahmeer Ahmad, Mohamed Eltabakh, Mourad Ouzzani, Nan Tang. *VLDB Endowment 2024*. [[pdf](https://dl.acm.org/doi/10.14778/3685800.3685890)]



#### 4.1.2 LLM for Data Integration

##### Entity Matching

1. MatchGPT: **Entity matching using large language models**  
   Ralph Peeters, Christian Bizer. *EDBT 2025.* [[pdf](https://arxiv.org/abs/2310.11244)]
2. COMEM: **Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching**  
   Tianshu Wang, Hongyu Lin, Xiaoyang Chen, Xianpei Han, Hao Wang, Zhenyu Zeng, Le Sun. *COLING 2025.* [[pdf](https://aclanthology.org/2025.coling-main.8/)]
3. BATCHER: **Cost-Effective In-Context Learning for Entity Resolution: A Design Space Exploration**  
   Meihao Fan, Xiaoyue Han, Ju Fan, Chengliang Chai, Nan Tang, Guoliang Li, Xiaoyong Du. *ICDE 2024.* [[pdf](https://ieeexplore.ieee.org/document/10597751)]
4. KcMF: **KcMF: A Knowledge-compliant Framework for Schema and Entity Matching with Fine-tuning-free LLMs**  
   Yongqin Xu, Huan Li, Ke Chen, Lidan Shou. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2410.12480)]
5. Jellyfish: **Jellyfish: A Large Language Model for Data Preprocessing**  
   Haochen Zhang, Yuyang Dong, Chuan Xiao, Masafumi Oyamada. *EMNLP 2024.* [[pdf](https://arxiv.org/abs/2312.01678)]

##### Schema Matching

1. KG-RAG4SM: **Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching**  
   *Chuangtao Ma, Sriom Chakrabarti, Arijit Khan, Bálint Molnár*. *arxiv 2025. [[pdf](https://arxiv.org/abs/2501.08686)]*
2. Harmonia: **Interactive Data Harmonization with LLM Agents**  
   Aécio Santos, Eduardo H. M. Pena, Roque Lopez, Juliana Freire. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2502.07132)]
3. LLMSchemaBench: **Schema Matching with Large Language Models: an Experimental Study**  
   Marcel Parciak, Brecht Vandevoort, Frank Neven, Liesbet M. Peeters, Stijn Vansummeren. *TaDA 2024 Workshop, collocated with VLDB 2024.* [[pdf](https://doi.org/10.48550/arXiv.2407.11852)]
4. Magneto: **Magneto: Combining Small and Large Language Models for Schema Matching**  
   Yurong Liu, Eduardo Pena, Aecio Santos, Eden Wu, Juliana Freire. *VLDB Endowment 2024.* [[pdf](https://doi.org/10.48550/arXiv.2412.08194)] [[pvldb](https://www.vldb.org/pvldb/vol17/p2750-fan.pdf)]
5. Agent-OM: **Agent-OM: Leveraging LLM Agents for Ontology Matching**
   Zhangcheng Qiang, Weiqing Wang, Kerry Taylor. *Proceedings of the VLDB Endowment, Volume 18, Issue 3, 2024.* [[pdf](https://dl.acm.org/doi/10.14778/3712221.3712222)]



#### 4.1.3 LLM for Data Discovery

1. ArcheType: **ArcheType: A Novel Framework for Open-Source Column Type Annotation using Large Language Models**  
   Benjamin Feuer, Yurong Liu, Chinmay Hegde, Juliana Freire. *VLDB 2024*. [[pdf](https://arxiv.org/abs/2310.18208#:~:text=We%20introduce%20ArcheType%2C%20a%20simple%2C%20practical%20method%20for,solve%20CTA%20problems%20in%20a%20fully%20zero-shot%20manner.)]

##### Data Profiling

1. Pneuma: **Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System**  
   Muhammad Imam Luthfi Balaka, David Alexander, Qiming Wang, Yue Gong, Adila Krisnadhi, Raul Castro Fernandez. *SIGMOD 2025*. [[pdf](https://arxiv.org/abs/2504.09207#:~:text=In%20this%20paper%2C%20we%20introduce%20Pneuma%2C%20a%20retrieval-augmented,designed%20to%20efficiently%20and%20effectively%20discover%20tabular%20data.)]
2. AutoDDG: **AutoDDG: Automated Dataset Description Generation using Large Language Models**  
   Haoxiang Zhang, Yurong Liu, Wei-Lun (Allen) Hung, Aécio Santos, Juliana Freire. *arxiv 2025.* [[pdf](https://arxiv.org/abs/2502.01050)]
3. LEDD: **LEDD: Large Language Model-Empowered Data Discovery in Data Lakes**  
   Qi An, Chihua Ying, Yuqing Zhu, Yihao Xu, Manwei Zhang, Jianmin Wang. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2502.15182)]

##### Data Annotation

1. Birdie: **Birdie: Natural Language-Driven Table Discovery Using Differentiable Search Index**  
   Yuxiang Guo, Zhonghao Hu, Yuren Mao, Baihua Zheng, Yunjun Gao, Mingwei Zhou. *VLDB 2025*. [[pdf](https://arxiv.org/abs/2504.21282)]
2. Goby: **Mind the Data Gap: Bridging LLMs to Enterprise Data Integration**  
   Moe Kayali, Fabian Wenz, Nesime Tatbul, Çağatay Demiralp. *CIDR 2025.* [[pdf](https://arxiv.org/abs/2412.20331)]
3. LLMCTA: **Evaluating Knowledge Generation and Self-Refinement Strategies for LLM-based Column Type Annotation**  
   Keti Korini, Christian Bizer. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2503.02718)]
4. CHORUS: **CHORUS: Foundation Models for Unified Data Discovery and Exploration**  
   Moe Kayali, Anton Lykov, Ilias Fountalis, Nikolaos Vasiloglou, Dan Olteanu, Dan Suciu. *Proceedings of the VLDB Endowment, Volume 17, Issue 8, 2024.* [[pdf](https://dl.acm.org/doi/10.14778/3659437.3659461)]
5. RACOON: **RACOON: An LLM-based Framework for Retrieval-Augmented Column Type Annotation with a Knowledge Graph**  
   Lindsey Linxi Wei, Guorui Xiao, Magdalena Balazinska. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2409.14556)]



### 4.2 LLM for Data Analysis

[⬆️top](#table-of-contents)

#### 4.2.1 LLM for Structured Data Analysis

1. **Survey of graph database models**  
   Renzo Angles, Claudio Gutierrez. *ACM Computing Surveys (CSUR), Volume 40, Issue 1, 2008*. [[pdf](https://dl.acm.org/doi/10.1145/1322432.1322433)]
2. **A Relational Model of Data for Large Shared Data Banks**   
   E. F. Codd. *Communications of the ACM 1970.* [[pdf](https://doi.org/10.1145/362384.362685)]

##### 4.2.1.1 Relational Data Analysis

###### LLM for Natural Language Interfaces

1. **Cracking SQL Barriers: An LLM-based Dialect Translation System**  
   Wei Zhou, Yuyang Gao, Xuanhe Zhou, Guoliang Li. *SIGMOD 2025*. [[pdf](https://dbgroup.cs.tsinghua.edu.cn/ligl/SIGMOD25-CrackSQL.pdf)]
2. **CrackSQL: A Hybrid SQL Dialect Translation System Powered by Large Language Models**    
   Wei Zhou, Yuyang Gao, Xuanhe Zhou, Guoliang Li. *arXiv 2025*. [[pdf](https://arxiv.org/abs/2504.00882#:~:text=In%20this%20demonstration%2C%20we%20present%20CrackSQL%2C%20the%20first,rule%20and%20LLM-based%20methods%20to%20overcome%20these%20limitations.)]
3. **FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis**  
   Chao Zhang, Yuren Mao, Yijiang Fan, Yu Mi, Yunjun Gao, Lu Chen, Dongfang Lou, Jinshu Lin. *SIGMOD 2024.* [[pdf](https://doi.org/10.1145/3626246.3653375)]
4. CodeS: **CodeS: Towards Building Open-source Language Models for Text-to-SQL**  
   Haoyang Li, Jing Zhang, Hanbing Liu, Ju Fan, Xiaokang Zhang, Jun Zhu, Renjie Wei, Hongyan Pan, Cuiping Li, Hong Chen. *Proceedings of the ACM on Management of Data, Volume 2, Issue 3, 2024.* [[pdf](https://doi.org/10.1145/3654930)]
5. **The Dawn of Natural Language to SQL: Are We Fully Ready?**  
   Boyan Li, Yuyu Luo, Chengliang Chai, Guoliang Li, Nan Tang. *VLDB 2024.* [[pdf](https://arxiv.org/abs/2406.01265)]
6. DataCoder: **Contextualized Data-Wrangling Code Generation in Computational Notebooks**  
   Junjie Huang, Daya Guo, Chenglong Wang, Jiazhen Gu, Shuai Lu, Jeevana Priya Inala, Cong Yan, Jianfeng Gao, Nan Duan, Michael R. Lyu. *ASE 2024*. [[pdf](https://dl.acm.org/doi/abs/10.1145/3691620.3695503)]
7. **PET-SQL: A Prompt-Enhanced Two-Round Refinement of Text-to-SQL with Cross-consistency**  
   Zhishuai Li, Xiang Wang, Jingjing Zhao, Sun Yang, Guoqing Du, Xiaoru Hu, Bin Zhang, Yuxiao Ye, Ziyue Li, Rui Zhao, Hangyu Mao. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.09732)]
8. CHESS: **CHESS: Contextual Harnessing for Efficient SQL Synthesis**  
   Shayan Talaei, Mohammadreza Pourreza, Yu-Chen Chang, Azalia Mirhoseini, Amin Saberi. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2405.16755)]
9. Data Interpreter:**Data Interpreter: An LLM Agent For Data Science**  
   Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, et al. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2402.18679)]
10. **DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction**  
    Mohammadreza Pourreza, Davood Rafiei. *NeurIPS 2023*. [[pdf](https://dl.acm.org/doi/10.5555/3666122.3667699)]
11. PACHINCO: **Natural Language to Code Generation in Interactive Data Science Notebooks**   
    Pengcheng Yin, Wen-Ding Li, Kefan Xiao, Abhishek Rao, Yeming Wen, Kensen Shi, Joshua Howland, Paige Bailey, Michele Catasta, Henryk Michalewski, Oleksandr Polozov, Charles Sutton. *ACL 2023.* [[pdf](https://aclanthology.org/2023.acl-long.9/)]
12. PALM: **PaLM: Scaling Language Modeling with Pathways**   
    Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al. *JMLR 2023.* [[pdf](https://dl.acm.org/doi/10.5555/3648699.3648939)]

###### LLM for Semantic Analysis

1. TableMaster: **TableMaster: A Recipe to Advance Table Understanding with Language Models**  
   Lang Cao. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.19378)]
2. Extractor-Reasoner-Executor paradigm: **TAT-LLM: A Specialized Language Model for Discrete Reasoning over Financial Tabular and Textual Data**  
   Fengbin Zhu, Ziyang Liu, Fuli Feng, Chao Wang, Moxin Li, Tat Seng Chua. *ICAIF 2024.* [[pdf](https://doi.org/10.1145/3677052.3698685)]
3. CABINET: **CABINET: Content Relevance based Noise Reduction for Table Question Answering**  
   Sohan Patnaik, Heril Changwal, Milan Aggarwal, Sumit Bhatia, Yaman Kumar, Balaji Krishnamurthy. *ICLR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.01155)]
4. Table-LLaVA: **Multimodal Table Understanding**  
   Mingyu Zheng, Xinwei Feng, Qingyi Si, Qiaoqiao She, Zheng Lin, Wenbin Jiang, Weiping Wang. *ACL 2024*. [[pdf](https://aclanthology.org/2024.acl-long.493/)]
5. TabPedia: **TabPedia: Towards Comprehensive Visual Table Understanding with Concept Synergy**  
   Weichao Zhao, Hao Feng, Qi Liu, Jingqun Tang, Shu Wei, Binghong Wu, Lei Liao, Yongjie Ye, Hao Liu, Wengang Zhou, Houqiang Li, Can Huang. *NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2406.01326)]
6. TAPERA: **TaPERA: Enhancing Faithfulness and Interpretability in Long-Form Table QA by Content Planning and Execution-based Reasoning**  
   Yilun Zhao, Lyuhao Chen, Arman Cohan, Chen Zhao. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.692/)]
7. ReAcTable: **ReAcTable: Enhancing ReAct for Table Question Answering**  
   Yunjia Zhang, Jordan Henkel, Avrilia Floratou, Joyce Cahoon, Shaleen Deep, Jignesh M. Patel. *Proceedings of the VLDB Endowment, Volume 17, Issue 8, 2024.* [[pdf](https://doi.org/10.14778/3659437.3659452)]
8. CHAIN-OF-TABLE: **Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding**  
   Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee, Tomas Pfister. *ICLR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2401.04398)]
9. TableGPT: **Table-GPT: Table Fine-tuned GPT for Diverse Table Tasks**  
   Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle Rifinski Fainman, Dongmei Zhang, Surajit Chaudhuri. *Proceedings of the ACM on Management of Data, Volume 2, Issue 3, 2024*. [[pdf](https://dl.acm.org/doi/10.1145/3654979)]
10. Qwen2.5: **Qwen2.5 Technical Report**  
    Qwen Team. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2412.15115)]
11. TableGPT2: **TableGPT2: A Large Multimodal Model with Tabular Data Integration**  
    Aofeng Su, Aowen Wang, Chao Ye, Chen Zhou, et al. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2411.02059)]
12. S3HQA: **S3HQA: A Three-Stage Approach for Multi-hop Text-Table Hybrid Question Answering**   
    Fangyu Lei, Xiang Li, Yifan Wei, Shizhu He, Yiming Huang, Jun Zhao, Kang Liu. *ACL 2023.* [[pdf](https://aclanthology.org/2023.acl-short.147/)]
13. Vicuna-7B: **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**   
    Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. *NeurIPS 2023 Datasets and Benchmarks Track.* [[pdf](https://doi.org/10.48550/arXiv.2306.05685)]



##### 4.2.1.2 Graph Data Analysis

1. **A Comparison of Current Graph Database Models**   
   Renzo Angles. *ICDEW 2012.* [[pdf](https://doi.org/10.1109/ICDEW.2012.31)]
2. Blazegraph: [[source](https://blazegraph.com/)]
3. GraphDB: [[source](https://graphdb.ontotext.com/)]
4. Neo4j: [[Github](https://github.com/neo4j/neo4j)]

###### Natural Language To Graph Analysis Query

1. r3-NL2GQL: **R3-NL2GQL: A Model Coordination and Knowledge Graph Alignment Approach for NL2GQL**   
   Yuhang Zhou, Yu He, Siyu Tian, Yuchen Ni, Zhangyue Yin, Xiang Liu, Chuanjun Ji, Sen Liu, Xipeng Qiu, Guangnan Ye, Hongfeng Chai. *Findings of EMNLP 2024.* [[pdf](https://aclanthology.org/2024.findings-emnlp.800/)]
2. **NAT-NL2GQL: A Novel Multi-Agent Framework for Translating Natural Language to Graph Query Language**  
   Yuanyuan Liang, Tingyu Xie, Gan Peng, Zihao Huang, Yunshi Lan, Weining Qian. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2412.10434)]
3. **Graph Learning in the Era of LLMs: A Survey from the Perspective of Data, Models, and Tasks**  
   Xunkai Li, Zhengyu Wu, Jiayi Wu, Hanwen Cui, Jishuo Jia, Rong-Hua Li, Guoren Wang. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2412.12456)]
4. **Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey**  
   Qizhi Pei, Lijun Wu, Kaiyuan Gao, Jinhua Zhu, Yue Wang, Zun Wang, Tao Qin, Rui Yan. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2403.01528)]

###### LLM-based Semantic Analysis

1. GraphGPT: **GraphGPT: Graph Instruction Tuning for Large Language Models**   
   Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, Chao Huang. *SIGIR 2024.* [[pdf](https://doi.org/10.48550/arXiv.2310.13023)]
2. Interactive-KBQA: **Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models**   
   Guanming Xiong, Junwei Bao, Wen Zhao. *ACL 2024.* [[pdf](https://aclanthology.org/2024.acl-long.569/)]
3. FlexKBQA: **FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering**     
   Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, Jianyong Wang. *AAAI 2024.* [[pdf](https://doi.org/10.48550/arXiv.2308.12060)]
4. InstructGLM: **Language is All a Graph Needs**   
   Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, Yongfeng Zhang. *EACL 2024.* [[pdf](https://aclanthology.org/2024.findings-eacl.132/)]
5. InstructGraph: **InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment**   
   Jianing Wang, Junda Wu, Yupeng Hou, Yao Liu, Ming Gao, Julian McAuley. *Findings of ACL 2024.* [[pdf](https://aclanthology.org/2024.findings-acl.801/)]
6. Readi: **Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments**  
   Sitao Cheng, Ziyuan Zhuang, Yong Xu, Fangkai Yang, Chaoyun Zhang, Xiaoting Qin, Xiang Huang, Ling Chen, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, Qi Zhang. *Findings of ACL 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.08593)]
7. Direct Preference Optimization (DPO) algorithm: **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**   
   Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn. *NeurIPS 2023.* [[pdf](https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)]
8. StructGPT: **StructGPT: A General Framework for Large Language Model to Reason over Structured Data**   
   Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, Ji-Rong Wen. *EMNLP 2023.* [[pdf](https://doi.org/10.48550/arXiv.2305.09645)]
9. UniKGQA: **UniKGQA: Unified Retrieval and Reasoning for Solving Multi-hop Question Answering Over Knowledge Graph**   
   Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen. *ICLR 2023.* [[pdf](https://doi.org/10.48550/arXiv.2212.00959)]
10. **Subgraph Retrieval Enhanced Model for Multi-hop Knowledge Base Question Answering**   
    Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, Hong Chen. *ACL 2022.* [[pdf](https://aclanthology.org/2022.acl-long.396/)]
11. RoBERTa: **RoBERTa: A Robustly Optimized BERT Pretraining Approach**  
    Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. *arXiv 2019*. [[pdf](https://arxiv.org/abs/1907.11692)]
12. Graph-SAGE: **Inductive representation learning on large graphs**   
    William L. Hamilton, Rex Ying, Jure Leskovec. *NeurIPS 2017.* [[pdf](https://dl.acm.org/doi/10.5555/3294771.3294869)]
13. GCN:**Semi-Supervised Classification with Graph Convolutional Networks**   
    Thomas N. Kipf, Max Welling. *ICLR 2017.* [[pdf](https://doi.org/10.48550/arXiv.1609.02907)]



#### 4.2.2 LLM for Semi-Structured Data Analysis

1. **Querying Semi-Structured Data**  
   Serge Abiteboul. *ICDT 1997.* [[pdf](https://dl.acm.org/doi/10.5555/645502.656103)]

##### 4.2.2.1 Markup Language

##### 4.2.2.2 Semi-Structured Tables

1. MiMoTable: **MiMoTable: A Multi-scale Spreadsheet Benchmark with Meta Operations for Table Reasoning**  
   Zheng Li, Yang Du, Mao Zheng, Mingyang Song. *COLING 2025.* [[pdf](https://doi.org/10.48550/arXiv.2412.11711)]
2. SPREADSHEETBENCH: **SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation**  
   Zeyao Ma, Bohan Zhang, Jing Zhang, Jifan Yu, Xiaokang Zhang, Xiaohan Zhang, Sijia Luo, Xi Wang, Jie Tang. *NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2406.14991)]
3. TEMPTABQA: **TempTabQA: Temporal Question Answering for Semi-Structured Tables**  
   Vivek Gupta, Pranshu Kandoi, Mahek Bhavesh Vora, Shuo Zhang, Yujie He, Ridho Reinanda, Vivek Srikumar. *EMNLP 2023.* [[pdf](https://doi.org/10.48550/arXiv.2311.08002)]



#### 4.2.3 LLM for Unstructured Data Analysis

##### 4.2.3.1 Documents

1. DocFormerV2: **DocFormerv2: Local Features for Document Understanding**  
   Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, R. Manmatha. *AAAI 2024.* [[pdf](https://doi.org/10.1609/aaai.v38i2.27828)]
2. mPLUG-DocOwl1.5: **mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding**  
   Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Ji Zhang, Qin Jin, Fei Huang, Jingren Zhou. *Findings of EMNLP 2024.* [[pdf](https://aclanthology.org/2024.findings-emnlp.175/)]
3. DocPedia: **DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain for Versatile Document Understanding**  
   Hao Feng, Qi Liu, Hao Liu, Jingqun Tang, Wengang Zhou, Houqiang Li, Can Huang. *SCIS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2311.11810)]
4. **Focus Anywhere for Fine-grained Multi-page Document Understanding**  
   Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, Xiangyu Zhang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2405.14295)]
5. **General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model**  
   Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, Chunrui Han, Xiangyu Zhang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2409.01704v1)]
6. DUBLIN: **DUBLIN: Visual Document Understanding By Language-Image Network**  
   Kriti Aggarwal, Aditi Khandelwal, Kumar Tanmay, Owais Khan Mohammed, Qiang Liu, Monojit Choudhury, Hardik Chauhan, Subhojit Som, Vishrav Chaudhary, Saurabh Tiwary. *EMNLP Industry Track 2023.* [[pdf](https://aclanthology.org/2023.emnlp-industry.65/)]
7. Pix2Struct: **Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding**  
   Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova. *ICML 2023.* [[pdf](https://dl.acm.org/doi/10.5555/3618408.3619188?ref=localhost)]
8. UDOP: **Unifying Vision, Text, and Layout for Universal Document Processing**  
   Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, Mohit Bansal. *CVPR 2023.* [[pdf](https://arxiv.org/abs/2212.02623v3)]
9. T5: **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**  
   Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. *JMLR 2020.* [[pdf](https://dl.acm.org/doi/10.5555/3455716.3455856)]
10. ViT: **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**  
    Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. *ICLR  2021.* [[pdf](https://iclr.cc/virtual/2021/oral/3458)]
11. JPEG DCT: **The JPEG Still Picture Compression Standard**  
    Gregory K. Wallace. *Communications of the ACM 1991.* [[pdf](https://doi.org/10.1145/103085.103089)]



##### 4.2.3.2 Program Language Analysis

###### LLM as Program Vulnerability Detection Tools

1. PDBER: **Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks**  
   Zhongxin Liu, Zhijie Tang, Junwei Zhang, Xin Xia, Xiaohu Yang. *ICSE 2024.* [[pdf](https://doi.org/10.1145/3597503.3639142)]
2. **Large Language Model for Vulnerability Detection: Emerging Results and Future Directions**  
   Xin Zhou, Ting Zhang, David Lo. *ICSE-NIER 2024.* [[pdf](https://doi.org/10.1145/3639476.3639762)]
3. control flow graph (CFG): **Vulnerability Detection by Learning From Syntax-Based Execution Paths of Code**  
   Junwei Zhang, Zhongxin Liu, Xing Hu, Xin Xia, Shanping Li. *IEEE TSE 2023.* [[pdf](https://ieeexplore.ieee.org/document/10153647)]
4. VUL-GPT: **Software Vulnerability Detection with GPT and In-Context Learning**  
   Zhihong Liu, Qing Liao, Wenchao Gu, Cuiyun Gao. *DSC 2023.* [[pdf](https://ieeexplore.ieee.org/abstract/document/10381286)]
5. CodeBERT: **CodeBERT: A Pre-Trained Model for Programming and Natural Languages**  
   Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, Ming Zhou. *Findings of EMNLP 2020.* [[pdf](https://aclanthology.org/2020.findings-emnlp.139/)]
6. BM25: **The Probabilistic Relevance Framework: BM25 and Beyond**  
   Stephen Robertson, Hugo Zaragoza. *Foundations and Trends in Information Retrieval, Volume 3, Issue 4, 2009.* [[pdf](https://dl.acm.org/doi/10.1561/1500000019)]

###### LLM-based Semantic-aware Analysis

1. ASTs: **Improving Code Summarization With Tree Transformer Enhanced by Position-Related Syntax Complement**  
   Jie Song, Zexin Zhang, Zirui Tang, Shi Feng, Yu Gu. *IEEE TAI 2024.* [[pdf](https://ieeexplore.ieee.org/document/10510878/metrics#metrics)]
2. **Large Language Models are Few-Shot Summarizers: Multi-Intent Comment Generation via In-Context Learning**  
   Mingyang Geng, Shangwen Wang, Dezun Dong, Haotian Wang, Ge Li, Zhi Jin, Xiaoguang Mao, Xiangke Liao. *ICSE 2024.* [[pdf](https://doi.org/10.1145/3597503.3608134)]
3. **Automatic Semantic Augmentation of Language Model Prompts (for Code Summarization)**  
   Toufique Ahmed, Kunal Suresh Pai, Premkumar Devanbu, Earl Barr. *ICSE 2024.* [[pdf](https://doi.org/10.1145/3597503.3639183)]
4. CoCoMIC: **CoCoMIC: Code Completion by Jointly Modeling In-file and Cross-file Context**  
   Yangruibo Ding, Zijian Wang, Wasi Ahmad, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang. *LREC-COLING 2024.* [[pdf](https://aclanthology.org/2024.lrec-main.305/)]
5. Repoformer: **Repoformer: Selective Retrieval for Repository-Level Code Completion**  
   Di Wu, Wasi Uddin Ahmad, Dejiao Zhang, Murali Krishna Ramanathan, Xiaofei Ma. *ICML 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.10059)]
6. SCLA: **SCLA: Automated Smart Contract Summarization via LLMs and Semantic Augmentation**  
   Yingjie Mao, Xiaoqi Li, Wenkai Li, Xin Wang, Lei Xie. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2402.04863)]
7. **Code Structure–Guided Transformer for Source Code Summarization**  
   Shuzheng Gao, Cuiyun Gao, Yulan He, Jichuan Zeng, Lunyiu Nie, Xin Xia, Michael Lyu. *ACM Transactions on Software Engineering and Methodology 2023.* [[pdf](https://doi.org/10.1145/3522674)]
8. RepoFusion: **RepoFusion: Training Code Models to Understand Your Repository**  
   Disha Shrivastava, Denis Kocetkov, Harm de Vries, Dzmitry Bahdanau, Torsten Scholak. *arXiv 2023.* [[pdf](https://doi.org/10.48550/arXiv.2306.10998)]



### 4.3 LLM for Data System Optimization

[⬆️top](#table-of-contents)

#### 4.3.1 LLM for Configuration Tuning

1. Breaking It Down: **Breaking It Down: An In-Depth Study of Index Advisors**  
   Wei Zhou, Chen Lin, Xuanhe Zhou, Guoliang Li. *Proceedings of the VLDB Endowment, Volume 17, Issue 10. 2024.*. [[pdf](https://dl.acm.org/doi/10.14778/3675034.3675035)]
2. TRAP: **TRAP: Tailored Robustness Assessment for Index Advisors via Adversarial Perturbation**  
   Wei Zhou; Chen Lin; Xuanhe Zhou; Guoliang Li; Tianqing Wang. *2024 IEEE 40th International Conference on Data Engineering (ICDE)*. [[pdf](https://ieeexplore.ieee.org/document/10597867)]
3. **Automatic Database Knob Tuning: A Survey**  
   Xinyang Zhao, Xuanhe Zhou, Guoliang Li. *IEEE Transactions on Knowledge and Data Engineering, Volume 35, Issue 12. 2023.*  [[pdf](https://dl.acm.org/doi/10.1109/TKDE.2023.3266893)]
4. **Demonstration of ViTA: Visualizing, Testing and Analyzing Index Advisors **    
   Wei Zhou, Chen Lin, Xuanhe Zhou, Guoliang Li, Tianqing Wang. *CIKM 2023.* [[pdf](https://dl.acm.org/doi/abs/10.1145/3583780.3614738)]
5. **An Efficient Transfer Learning Based Configuration Adviser for Database Tuning**  
   Xinyi Zhang, Hong Wu, Yang Li, Zhengju Tang, Jian Tan, Feifei Li, Bin Cui. *Proceedings of the VLDB Endowment, Volume 17, Issue 3. 2023.* [[pdf](https://dl.acm.org/doi/abs/10.14778/3632093.3632114)]
6. **Code-aware cross-program transfer hyperparameter optimization**  
   Zijia Wang, Xiangyu He, Kehan Chen, Chen Lin, Jinsong Su. *AAAI 2023.* [[pdf](https://dl.acm.org/doi/10.1609/aaai.v37i9.26226)]
7. QTune: **QTune: a query-aware database tuning system with deep reinforcement learning**  
   Guoliang Li, Xuanhe Zhou, Shifu Li, Bo Gao. *Proceedings of the VLDB Endowment, Volume 12, Issue 12. 2019.* [[pdf](https://dl.acm.org/doi/10.14778/3352063.3352129)]

##### Tuning Task-Aware Prompt Engineering

1. λ-Tune: **λ-Tune: Harnessing Large Language Models for Automated Database System Tuning**  
   Victor Giannankouris, Immanuel Trummer. *SIGMOD 2025.* [[pdf](https://doi.org/10.48550/arXiv.2411.03500)]
2. LLMIdxAdvis: **LLMIdxAdvis: Resource-Efficient Index Advisor Utilizing Large Language Model**  
   Xinxin Zhao, Haoyang Li, Jing Zhang, Xinmei Huang, Tieying Zhang, Jianjun Chen, Rui Shi, Cuiping Li, Hong Chen. *arxiv 2025.* [[pdf](https://arxiv.org/abs/2503.07884)]
3. LATuner: **LATuner: An LLM-Enhanced Database Tuning System Based on Adaptive Surrogate Model**  
   Chongjiong Fan, Zhicheng Pan, Wenwen Sun, Chengcheng Yang, Wei-Neng Chen. *ECML PKDD 2024.* [[pdf](https://doi.org/10.1007/978-3-031-70362-1_22)]
4. LLMBench: **Is Large Language Model Good at Database Knob Tuning? A Comprehensive Experimental Evaluation**  
   Yiyan Li, Haoyang Li, Zhao Pu, Jing Zhang, Xinyi Zhang, Tao Ji, Luming Sun, Cuiping Li, Hong Chen. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2408.02213)]

##### RAG Based Tuning Experience Enrichment

1. Andromeda: **Automatic Database Configuration Debugging using Retrieval-Augmented Language Models**  
   Sibei Chen, Ju Fan, Bin Wu, Nan Tang, Chao Deng, Pengyi Wang, Ye Li, Jian Tan, Feifei Li, Jingren Zhou, Xiaoyong Du. *Proceedings of the ACM on Management of Data, Volume 3, Issue 1, 2025.* [[pdf](https://dl.acm.org/doi/10.1145/3709663)]
2. GPTuner: **GPTuner: A Manual-Reading Database Tuning System via GPT-Guided Bayesian Optimization**  
   Jiale Lao, Yibo Wang, Yufei Li, Jianping Wang, Yunjia Zhang, Zhiyuan Cheng, Wanghu Chen, Mingjie Tang, Jianguo Wang. *VLDB 2024.* [[pdf](https://doi.org/10.14778/3659437.3659449)]

##### Training Enhanced Tuning Goal Alignment

1. E2ETune: **E2ETune: End-to-End Knob Tuning via Fine-tuned Generative Language Model**  
   Xinmei Huang, Haoyang Li, Jing Zhang, Xinxin Zhao, Zhiming Yao, Yiyan Li, Tieying Zhang, Jianjun Chen, Hong Chen, Cuiping Li. *VLDB 2025.* [[pdf](https://doi.org/10.48550/arXiv.2404.11581)]
2. DB-GPT: **DB-GPT: Large Language Model Meets Database**  
   Xuanhe Zhou, Zhaoyan Sun, Guoliang Li. *Data Science and Engineering 2024.* [[pdf](https://link.springer.com/article/10.1007/s41019-023-00235-6)]
3. HEBO algorithm: **HEBO: Heteroscedastic Evolutionary Bayesian Optimisation**  
   Alexander I. Cowen-Rivers, Wenlong Lyu, Zhi Wang, Rasul Tutunov, Hao Jianye, Jun Wang, Haitham Bou Ammar. *NeurIPS 2020*. [[pdf](https://arxiv.org/abs/2012.03826v1)]



#### 4.3.2 LLM for Query Optimization

##### Optimization-Aware Prompt Engineering

1. LLM-QO: **Can Large Language Models Be Query Optimizer for Relational Databases?**  
   Jie Tan, Kangfei Zhao, Rui Li, Jeffrey Xu Yu, Chengzhi Piao, Hong Cheng, Helen Meng, Deli Zhao, Yu Rong. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.05562)]
2. LLMOpt: **A Query Optimization Method Utilizing Large Language Models**  
   Zhiming Yao, Haoyang Li, Jing Zhang, Cuiping Li, Hong Chen. *arxiv 2025.* [[pdf](https://arxiv.org/abs/2503.06902)]
3. LITHE: **Query Rewriting via LLMs**  
   Sriram Dharwada, Himanshu Devrani, Jayant Haritsa, Harish Doraiswamy. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.12918)]
4. DB-GPT: **DB-GPT: Large Language Model Meets Database**    
   Xuanhe Zhou, Zhaoyan Sun, Guoliang Li. *Data Science and Engineering 2024.* [[pdf](https://link.springer.com/article/10.1007/s41019-023-00235-6)]
5. LLM-R<sup>2</sup>: **LLM-R2: A Large Language Model Enhanced Rule-Based Rewrite System for Boosting Query Efficiency**  
   Zhaodonghui Li, Haitao Yuan, Huiming Wang, Gao Cong, Lidong Bing. *VLDB 2024.* [[pdf](https://doi.org/10.14778/3696435.3696440)]
6. LLMSteer: **The Unreasonable Effectiveness of LLMs for Query Optimization**  
   Peter Akioyamen, Zixuan Yi, Ryan Marcus. *ML for Systems Workshop at NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2411.02862)]
7. R-Bot: **R-Bot: An LLM-based Query Rewrite System**  
   Zhaoyan Sun, Xuanhe Zhou, Guoliang Li. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2412.01661)]
8. GenRewrite: **Query Rewriting via Large Language Models**  
   Jie Liu, Barzan Mozafari. *arXiv 2024.* [[pdf](https://doi.org/10.48550/arXiv.2403.09060)]



#### 4.3.3 LLM for Anomaly Diagnosis

##### Manually Crafted Prompts for Anomaly Diagnosis

1. DBG-PT: **DBG-PT: A Large Language Model Assisted Query Performance Regression Debugger**  
   Victor Giannakouris, Immanuel Trummer. *Proceedings of the VLDB Endowment, Volume 17, Issue 12, 2024.* [[pdf](https://doi.org/10.14778/3685800.3685869)]

##### RAG Based Diagnosis Experience Enrichment

1. ByteHTAP: **Query Performance Explanation through Large Language Model for HTAP Systems**   
   Haibo Xiu, Li Zhang, Tieying Zhang, Jun Yang, Jianjun Chen. *ICDE 2025.* [[pdf](https://doi.org/10.48550/arXiv.2412.01709)]
2. D-Bot: **D-Bot: Database Diagnosis System using Large Language Models**  
   Xuanhe Zhou, Guoliang Li, Zhaoyan Sun, Zhiyuan Liu, Weize Chen, Jianming Wu, Jiesi Liu, Ruohang Feng, Guoyang Zeng. *Proceedings of the VLDB Endowment, Volume 17, Issue 10. 2024.* [[pdf](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbot_vldb_camera_ready_v1.pdf)]
3. **LLM As DBA**  
   Xuanhe Zhou, Guoliang Li, Zhiyuan Liu. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2308.05481)]

##### Multi-Agent Mechanism for Collaborative Diagnosis

1. D-Bot: **D-Bot: Database Diagnosis System using Large Language Models**  
   Xuanhe Zhou, Guoliang Li, Zhaoyan Sun, Zhiyuan Liu, Weize Chen, Jianming Wu, Jiesi Liu, Ruohang Feng, Guoyang Zeng. *Proceedings of the VLDB Endowment, Volume 17, Issue 10. 2024.* [[pdf](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbot_vldb_camera_ready_v1.pdf)]
2. Panda: **Panda: Performance Debugging for Databases using LLM Agents**  
   Vikramank Singh, Kapil Eknath Vaidya, Vinayshekhar Bannihatti Kumar, Sopan Khosla, Balakrishnan Narayanaswamy, Rashmi Gangadharaiah, Tim Kraska. *CIDR 2024.* [[pdf](https://www.cidrdb.org/cidr2024/papers/p6-singh.pdf)]
3. **LLM As DBA**  
   Xuanhe Zhou, Guoliang Li, Zhiyuan Liu. *arXiv 2023.* [[pdf](https://arxiv.org/abs/2308.05481)]

##### Localized LLM Enhancement via Specialized FineTuning

1. D-Bot: **D-Bot: Database Diagnosis System using Large Language Models**  
   Xuanhe Zhou, Guoliang Li, Zhaoyan Sun, Zhiyuan Liu, Weize Chen, Jianming Wu, Jiesi Liu, Ruohang Feng, Guoyang Zeng. *Proceedings of the VLDB Endowment, Volume 17, Issue 10. 2024.* [[pdf](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/dbot_vldb_camera_ready_v1.pdf)]

##### Practices of LLMs for Data Management

1. Alibaba Cloud: [[source](https://bailian.console.aliyun.com/xiyan)]
2. Amazon Nova: [[source](https://aws.amazon.com/cn/ai/generative-ai/nova/understanding/)]
3. PawSQL: [[source](https://www.pawsql.com/)]
4. DBDoctor: [[source](https://www.dbdoctor.cn/)]



## 5 Challenges and Future Directions

[⬆️top](#table-of-contents)

### 5.1 Data Management for LLM

#### 5.1.1 Task-Specific Data Selection for Efficient Pretraining

#### 5.1.2 Optimizing Data Processing Pipelines

#### 5.1.3 LLM Knowledge Update and Version Control

#### 5.1.4 Comprehensive Dataset Evaluation

1. **Statistical Dataset Evaluation: Reliability, Difficulty, and Validity**  
   Chengwen Wang, Qingxiu Dong, Xiaochen Wang, Haitao Wang, Zhifang Sui. *arXiv 2022*. [[pdf](https://arxiv.org/abs/2212.09272)]

#### 5.1.5 Hybrid RAG Indexing and Retrieval

1. Elasticsearch: [[source](https://www.elastic.co/elasticsearch)]
2. re-ranking: **ASRank: Zero-Shot Re-Ranking with Answer Scent for Document Retrieval**  
   Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Adam Jatowt. *NAACL 2025.* [[pdf](https://doi.org/10.48550/arXiv.2501.15245)]
3. LightRAG: **LightRAG: Simple and Fast Retrieval-Augmented Generation**  
   Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2410.05779)]
4. AutoRAG: **AutoRAG: Automated Framework for optimization of Retrieval Augmented Generation Pipeline**  
   Dongkyu Kim, Byoungwook Kim, Donggeon Han, Matouš Eibich. *arXiv 2024*. [[pdf](https://arxiv.org/abs/2410.20878)]



### 5.2 LLM for Data Management

[⬆️top](#table-of-contents)

#### 5.2.1 Unified Data Analysis System

#### 5.2.2 Data Analysis with Private Domain Knowledge

#### 5.2.3 Representing Non-Sequential and Non-Textual Data

1. λ-Tune: **λ-Tune: Harnessing Large Language Models for Automated Database System Tuning**  
   Victor Giannankouris, Immanuel Trummer. *SIGMOD 2025.* [[pdf](https://doi.org/10.48550/arXiv.2411.03500)]
2. LLMErrorBench: **Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets**  
   Tommaso Bendinelli, Artur Dox, Christian Holz. *ICLR 2025 Workshop on Foundation Models in the Wild*. [[pdf](https://arxiv.org/abs/2503.06664)]
3. LLM-QO: **Can Large Language Models Be Query Optimizer for Relational Databases?**  
   Jie Tan, Kangfei Zhao, Rui Li, Jeffrey Xu Yu, Chengzhi Piao, Hong Cheng, Helen Meng, Deli Zhao, Yu Rong. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.05562)]
4. LLMOpt: **A Query Optimization Method Utilizing Large Language Models**  
   Zhiming Yao, Haoyang Li, Jing Zhang, Cuiping Li, Hong Chen. *arxiv 2025.* [[pdf](https://arxiv.org/abs/2503.06902)]
5. RetClean: **RetClean: Retrieval-Based Data Cleaning Using LLMs and Data Lakes**  
   Zan Ahmad Naeem, Mohammad Shahmeer Ahmad, Mohamed Eltabakh, Mourad Ouzzani, Nan Tang. *Proceedings of the VLDB Endowment, Volume 17, Issue 12, 2024*. [[pdf](https://dl.acm.org/doi/10.14778/3685800.3685890)]
6. LATuner: **LATuner: An LLM-Enhanced Database Tuning System Based on Adaptive Surrogate Model**  
   Chongjiong Fan, Zhicheng Pan, Wenwen Sun, Chengcheng Yang, Wei-Neng Chen. *ECML PKDD 2024.* [[pdf](https://doi.org/10.1007/978-3-031-70362-1_22)]
7. LLMClean: **LLMClean: Context-Aware Tabular Data Cleaning via LLM-Generated OFDs**  
   Fabian Biester, Mohamed Abdelaal, Daniel Del Gaudio. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2404.18681)]
8. CleanAgent: **CleanAgent: Automating Data Standardization with LLM-based Agents**  
   Danrui Qi, Jiannan Wang. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2403.08291)]

#### 5.2.4 Efficient LLM Utilization Under Budget Constraints

1. LLM-QO: **Can Large Language Models Be Query Optimizer for Relational Databases?**  
   Jie Tan, Kangfei Zhao, Rui Li, Jeffrey Xu Yu, Chengzhi Piao, Hong Cheng, Helen Meng, Deli Zhao, Yu Rong. *arXiv 2025.* [[pdf](https://doi.org/10.48550/arXiv.2502.05562)]
2. GIDCL: **GIDCL: A Graph-Enhanced Interpretable Data Cleaning Framework with Large Language Models**  
   Mengyi Yan, Yaoshu Wang, Yue Wang, Xiaoye Miao, Jianxin Li. *Proceedings of the ACM on Management of Data, Volume 2, Issue 6, 2025.* [[pdf](https://dl.acm.org/doi/10.1145/3698811)]
3. LLM-R<sup>2</sup>: **LLM-R2: A Large Language Model Enhanced Rule-Based Rewrite System for Boosting Query Efficiency**  
   Zhaodonghui Li, Haitao Yuan, Huiming Wang, Gao Cong, Lidong Bing. *VLDB 2024.* [[pdf](https://doi.org/10.14778/3696435.3696440)]
4. LLMSteer: **The Unreasonable Effectiveness of LLMs for Query Optimization**  
   Peter Akioyamen, Zixuan Yi, Ryan Marcus. *ML for Systems Workshop at NeurIPS 2024.* [[pdf](https://doi.org/10.48550/arXiv.2411.02862)]
5. **Schema Matching with Large Language Models: an Experimental Study**  
   Marcel Parciak, Brecht Vandevoort, Frank Neven, Liesbet M. Peeters, Stijn Vansummeren. *TaDA 2024 Workshop, collocated with VLDB 2024.* [[pdf](https://doi.org/10.48550/arXiv.2407.11852)]
6. R-Bot: **R-Bot: An LLM-based Query Rewrite System**  
   Zhaoyan Sun, Xuanhe Zhou, Guoliang Li. *arXiv 2024.* [[pdf](https://arxiv.org/abs/2412.01661)]
