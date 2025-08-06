# CSV ile Semantic Search ve AI Agent Projesi

Bu proje, CSV dosyalarınızla doğal dil kullanarak etkileşim kurmanızı sağlayan bir AI agent'tır. Semantic search kullanarak CSV dosyanızdaki sütunları anlayıp, sorularınızı yanıtlar.

## Özellikler

- **Local AI Model**: Llama 3.1:8B modeli ile tamamen local çalışır
- **Semantic Search**: Nomic-embed-text-v1.5 embedding modeli ile güçlü semantic search
- **CSV Analizi**: CSV dosyalarınızı yükleyip analiz edebilirsiniz
- **Görselleştirme**: Verilerinizi grafiklerle görselleştirebilirsiniz
- **Doğal Dil Sorguları**: Türkçe veya İngilizce sorular sorabilirsiniz

## Kurulum

### 1. Ollama Kurulumu

Öncelikle Ollama'yı sisteminize kurmanız gerekiyor:

**Windows için:**
```bash
# https://ollama.ai adresinden indirin ve kurun
```

**macOS için:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Linux için:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Llama Modelini İndirin

Terminal'de aşağıdaki komutu çalıştırın:
```bash
ollama pull llama3.1:8b
```

### 3. Python Bağımlılıklarını Kurun

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Uygulamayı Başlatın

```bash
streamlit run main.py
```

### 2. CSV Dosyanızı Yükleyin

- Web arayüzünde "Upload a Excel File" butonuna tıklayın
- CSV dosyanızı seçin
- Dosya yüklendiğinde önizlemesini göreceksiniz

### 3. Sorularınızı Sorun

- "Type Here:" alanına sorularınızı yazın
- Örnek sorular:
  - "En yüksek değerler neler?"
  - "Ortalama değerleri göster"
  - "Bu veriyi grafikle görselleştir"
  - "En çok tekrar eden değerler neler?"

## Sistem Gereksinimleri

- **RAM**: En az 8GB (Llama 3.1:8B için)
- **Depolama**: En az 5GB boş alan
- **İşlemci**: Modern bir CPU (GPU opsiyonel)

## Sorun Giderme

### Ollama Bağlantı Hatası
Eğer "Ollama connection error" alıyorsanız:
1. Ollama'nın çalıştığından emin olun
2. Terminal'de `ollama serve` komutunu çalıştırın

### Model İndirme Hatası
Eğer model indirme hatası alıyorsanız:
1. İnternet bağlantınızı kontrol edin
2. `ollama pull llama3.1:8b` komutunu tekrar çalıştırın

### Bellek Hatası
Eğer bellek hatası alıyorsanız:
1. Diğer uygulamaları kapatın
2. Daha küçük bir model deneyin (örn: llama3.1:3b)

## Teknik Detaylar

- **Embedding Modeli**: nomic-ai/nomic-embed-text-v1.5
- **LLM Modeli**: llama3.1:8b (Ollama üzerinden)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Web Framework**: Streamlit

## Katkıda Bulunma

Bu proje açık kaynaklıdır. Katkılarınızı bekliyoruz!
