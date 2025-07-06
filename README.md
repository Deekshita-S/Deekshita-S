# 👋 Hi, I'm Deekshita!

🎓 **MSc Data Science Gold Medalist** (2025 Batch, DIAT)  

🔬 Research experience at IISER Bhopal (Federated Learning) and at GTRE,DRDO (Unsupervised Early failure detection)

💻 Passionate about Machine Learning, NLP, and Predictive Maintenance  

🧠 Always exploring new models, frameworks, and datasets

---

## 🧑‍💻 Projects

### 🏥 [Kenya Clinical Reasoning Challenge – MedAlpaca Fine-tuning](https://github.com/Deekshita-S/Clinical-reasoning---Kenya-Challenge)

- Fine-tuned a [MedAlpaca model on Hugging Face](https://huggingface.co/ink/kenya-clinical-medalpaca) using 400 real-world clinical prompts from Kenyan frontline healthcare settings.  
- Used **LoRA** (Low-Rank Adaptation) and **BitsAndBytes** quantization to efficiently fine-tune the model on a small dataset and limited compute.  
- Replaced corrupted characters in text using manual mappings and regex replacements to restore prompt clarity.  
- Extracted structured metadata (e.g., county, nursing experience, facility level, patient data) from both CSV and prompt content.  
- Used `bertscore` for validation as it aligned better with expert clinical judgment than ROUGE.  
- Best validation scores:

```json
{
  "eval_loss": 1.2426,
  "eval_bertscore_precision": 0.8871,
  "eval_bertscore_recall": 0.9182,
  "eval_bertscore_f1": 0.9023
}
```
### 💬 [Brand Sentiment Classification with BERT](https://github.com/Deekshita-S/Twitter-Brand-Sentiment-Analysis)
- Fine-tuned a BERT model on brand-related tweets for 3-class sentiment classification (positive, neutral, negative), using data augmentation and class-weighted loss to handle imbalance.
- Initial traditional ML approaches underperformed (~73% accuracy), prompting a shift to BERT-based modeling with optimized preprocessing strategies.
- Achieved ~87% accuracy and 88.6% F1-score, with performance tracked using Weights & Biases (WandB).
  

### 🛰️ [Communication-Efficient Federated Learning – IISER Bhopal Internship](https://github.com/Deekshita-S/LANDER)
- Modified the original LANDER code to reduce communication overhead and client training load in federated class-incremental learning.
- Introduced entropy-based image selection to send only the most informative 50% of synthetic images.
- Added a proximal term to the local loss function to improve model stability during client updates.
- Main changes made in lander.py (methods folder) and the ImagePool class.


### 🔧 [Degradation onset detection](https://github.com/Deekshita-S/Degradation-detection-in-CMAPSS-dataset)
- Developed a pipeline to detect degradation onset in the FD001 (CMAPSS) dataset using ML/DL models.
- Used DBSCAN clustering and t-SNE visualization to identify healthy segments in the sensor data.
- Trained unsupervised models (OCSVM, LSTM Autoencoders, Transformer Autoencoders) on healthy data only.
- Detected the onset of degradation based on reconstruction errors or decision function scores.



### ☀️ [Solar Panel Efficiency Prediction (ML Competition)](https://github.com/Deekshita-S/Solar-panel-efficiency-prediction---Zelestra-Challenge)
- Participated in the Zelestra X AWS ML Ascend Challenge, predicting solar panel efficiency using real-world sensor and metadata features.
- Performed extensive data cleaning, feature selection, and EDA, engineering domain-specific features and using IterativeImputer for missing values.
- Achieved a top-tier score (89.79/100)  using a stacked ensemble of Random Forest, XGBoost, and CatBoost, outperforming deep learning models


### 🚌 [Seat Demand Forecasting – Redbus Data Decode Hackathon](https://github.com/Deekshita-S/Redbus-challenge)

- Participated in the **Redbus Data Decode Hackathon**, predicting route-level seat bookings 15 days before the journey date using transaction and route metadata.  
- Engineered features based on **city tier, region pairs, holidays, and booking patterns**, while restricting training data to 15–30 days prior to journey.  
- Achieved **RMSE of 679**, ranking **#226 out of ~830 teams**, with **Random Forest** outperforming other models like XGBoost and LSTM.


---

## 🔧 Tools & Technologies

- **Languages:** Python, SQL  
- **Frameworks:** PyTorch, Hugging Face
- **Tools:** WandB, Git, Jupyter, Pandas, NumPy, Matplotlib, Seaborn, NLTK, scikit-learn
- **Specialties:** Data Augmentation, Domain Adaptation, Clustering, Transformer Models, NLP

---

## 📈 Currently Learning

- Hugging Face Transformers  
- Efficient Fine-tuning (LoRA, PEFT)  
- Advanced Model Monitoring & Evaluation  
- MLOps & Deployment Tools

---

## 📫 Connect With Me

- [LinkedIn](https://www.linkedin.com/in/deekshita-iyer-7554bb268/)  
- [deekshita809@gmail.com](mailto:deekshita809@gmail.com)

---

> “In data we trust, in learning we grow.”
