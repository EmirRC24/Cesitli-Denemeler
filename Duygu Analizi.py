import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Excel dosyasını yükleyin
df = pd.read_excel(r'C:/Users/emiri/OneDrive/Belgeler/GitHub/Cesitli-Denemeler/Örnek Veri Seti.xlsx')

# BERT modelini ve tokenizer'ı yükleyin
model_name = "savasy/bert-base-turkish-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Duygu analizi pipeline'ını oluşturun
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Yorumların olduğu sütunu seçiyoruz ve boş olmayanları alıyoruz
comments = df["Konuğumuzun oturumu hakkındaki düşünceleriniz"].dropna()

# Her bir yorum için duygu analizi yapıyoruz
sentiment_results = comments.apply(lambda x: sentiment_analysis(x)[0]['label'])

# Analiz sonuçlarını ana dataframe'e ekleyelim
df.loc[comments.index, "Duygu Analizi Sonucu"] = sentiment_results

# Sonuçları gösterelim
print(df[["Konuğumuzun oturumu hakkındaki düşünceleriniz", "Duygu Analizi Sonucu"]])

# Sonuçları bir Excel dosyasına kaydedin
df.to_excel(r'C:\Users\emiri\OneDrive\Belgeler\GitHub\Cesitli-Denemeler\Duygu_Analizi_Sonuclari.xlsx', index=False)
