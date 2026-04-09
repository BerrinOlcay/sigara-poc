import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

import os
from pypdf import PdfReader

@st.cache_data
def load_documents(folder_path):
    documents = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            reader = PdfReader(filepath)
            
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
               
                if any(keyword in page_text for keyword in [
                    "Overall questions",
                    "Justification and evidence",
                    "Recommendations",
                    "Implementation considerations"
                ]):
                    continue
                if len(page_text.strip()) < 100:
                    continue                

                # filtreleme
                if "Creative Commons" in page_text:
                    continue
                
                text += page_text
            
            documents.append(text)
    
    return documents

@st.cache_data
def split_text(documents, chunk_size=500):
    chunks = []
    
    for doc in documents:
        sentences = doc.split(".")
        
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "."
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks

def create_embeddings(chunks, client):
    embeddings = []
    
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    
    return embeddings

import numpy as np

def search(query, chunks, embeddings, client):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    
    similarities = []
    
    for emb in embeddings:
        similarity = np.dot(query_embedding, emb)
        similarities.append(similarity)
    
    # en yüksek benzerlik
    best_index = similarities.index(max(similarities))
    
    return chunks[best_index]

def etiket_cikar(metin, client):
    prompt = f"""
    Aşağıdaki hasta konuşmasını analiz et ve sadece JSON formatında etiketleri çıkar.
        
    {{
      "tetikleyiciler": [],
      "motivasyonlar": [],
      "guclukler": [],
      "birakma_gecmisi": [],
      "bagimlilik": ""
    }}
        
    Kullanılabilecek etiketler:
    
    tetikleyiciler: stres, kahve, cay, yemek_sonrasi, alkol, sosyal_ortam, yalnizlik, of$
    motivasyonlar: saglik, aile, cocuk, ekonomi
    guclukler: sinirlilik, uykusuzluk, asiri_istek
    birakma_gecmisi: onceki_deneme, relaps
    bagimlilik: dusuk, orta, yuksek
        
    Hasta konuşması:
    {metin}
    
    Sadece JSON döndür.
    JSON formatı aşağıdaki gibi olmalıdır:

    {{
      "tetikleyiciler": [],
      "motivasyonlar": [],
      "guclukler": [],
      "birakma_gecmisi": [],
      "bagimlilik": ""
    }}

    Kurallar:
    - Tüm alanlar DOLDURULMALIDIR, boş bırakma.
    - Eğer bilgi yoksa:
      - listeler için: []
      - bagimlilik için: "Belirtilmedi" yaz.
    - bagimlilik sadece şu değerlerden biri olmalıdır:
      "düşük", "orta", "yüksek", "Belirtilmedi"
    - Asla açıklama ekleme, sadece JSON döndür.
    """
        
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )   
    
    import json

    try:
        return json.loads(response.output_text)
    except:
        return {
            "tetikleyiciler": [],
            "motivasyonlar": [],
            "guclukler": [],
            "birakma_gecmisi": [],
            "bagimlilik": "Belirtilmedi"
    }

st.set_page_config(page_title="Klinik Karar Destek Sistemi", layout="centered")

st.title("🚬 Sigara Bırakma - Klinik Karar Destek Sistemi")
st.markdown("Bu sistem, hasta ifadelerini analiz ederek klinik karar desteği sağlar.")
st.caption("Privacy by Design: Bu PoC kapsamında girilen hasta verileri anonim olarak işlenmekte olup kalıcı olarak saklanmamaktadır.")

docs = load_documents("documents")
st.write(f"Yüklenen doküman sayısı: {len(docs)}")

chunks = split_text(docs)
st.write(f"Toplam parça sayısı: {len(chunks)}")



col1, col2, col3 = st.columns(3)
with col1:
    metin = st.text_area("Hasta Görüşmesi", height=250)

if st.button("Analiz Et"):
    embeddings=create_embeddings(chunks[:5], client)
    rag_context = search(metin, chunks, embeddings, client)
    etiketler = etiket_cikar(metin, client)
    st.markdown("### Hasta Profili")

    etiket = etiketler

    motivasyon = ", ".join(etiket.get("motivasyonlar", [])) or "Belirtilmedi"
    gucluk = ", ".join(etiket.get("guclukler", [])) or "Belirtilmedi"
    tetikleyici = ", ".join(etiket.get("tetikleyiciler", [])) or "Belirtilmedi"
    bagimlilik = etiket.get("bagimlilik") or "Belirtilmedi"

    st.write(f"*Motivasyon:* {motivasyon}")
    st.write(f"*Güçlük:* {gucluk}")
    st.write(f"*Tetikleyiciler:* {tetikleyici}")
    st.write(f"*Bağımlılık düzeyi:* {bagimlilik}")
    
    prompt = f"""
    Hasta ifadesinden çıkarılan etiketler:
    {etiketler}
    
Aşağıdaki hasta ifadesini analiz et ve sadece aşağıdaki formatta cevap ver.

Ambivalans: (Var / Yok)
Direnç: (Var / Yok)

Klinik Yorum:
(Kısa ve net 1-2 cümle. Hastanın ifadesindeki temel klinik durumu açıkça belirt.)

Öneri:
(Kısa, profesyonel ve klinik dilde yaz.
Öneri mutlaka hastanın ifadesindeki spesifik duruma DOĞRUDAN referans versin.
Genel tavsiye verme, kişiye özel klinik yönlendirme yap.
Öneri SOMUT ve UYGULANABİLİR olsun (örneğin: tetikleyici analizi, kısa hedef belirleme, nikotin replasman tedavisi gibi).
Kesin ve buyurgan dil kullanma.
Temkinli ve klinik öneri dili kullan ('önerilebilir', değerlendirilebilir', 'uygun olabilir').
Gerekirse davranışsal ve farmakolojik yaklaşımı birlikte belirt.
Öneri cümlelerinde klinik ve profesyonel ifade kullan.)

Kurallar:
- Etiketleri kullan
- Kısa ve net yaz
- Öneriyi soru şeklinde yazma
- Günlük dil kullanma, klinik ifade kullan
- Klinik yorum ve öneri, hastanın ifadesindeki duygusal veya davranışsal örüntüyü açıkça adlandırsın
- Öneri cümlelerinde doğrudan emir kipi kullanma, klinik öneri dili kullan (örneğin: "önerilir", "uygun olacaktır")

Rehber Bağlamı:
{rag_context}

Hasta ifadesi:
{metin}
"""
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    metin_cikti = response.output_text

    if 'Öneri:' in metin_cikti:
        analiz_kismi, oneri_kismi = metin_cikti.split('Öneri:', 1)
    else:
        analiz_kismi = metin_cikti
        oneri_kismi = ''

    with col2:
        st.subheader('Semantik Analiz')
        st.write(analiz_kismi)

    with col3:
        st.subheader('Öneri')
        st.write(oneri_kismi)
