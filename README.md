# YOK Akademik Analiz

Bu proje, YÖK Akademik Arama sistemi üzerinden belirli bir ana bilim dalına
ait akademisyenlerin makale/kitap üretkenliğini toplar, görselleştirir ve
unvan ile yayın hacmi arasındaki korelasyonu hesaplar.

## Gereksinimler

- Python 3.9+

## Kurulum

```bash
pip install -r requirements.txt
```

## Calistirma

```bash
python main.py --department "Bilgisayar Mühendisliği"
```

Varsayilan olarak arama terimi, bolum adinin ilk kelimesidir. Gerekirse
`--search-term` ile daha genel bir ifade verebilirsiniz.

```bash
python main.py --department "Bilgisayar Mühendisliği" --search-term "Bilgisayar"
```

### Opsiyonlar

- `--max-pages`: arama sayfasi siniri
- `--max-academics`: islenecek akademisyen sayisi
- `--sleep`: istekler arasi bekleme (saniye)
- `--insecure`: SSL dogrulamayi kapatir (yerel SSL hatalarinda)
- `--outdir`: cikti klasoru (grafikler + rapor)
- `--data-path`: akademisyen CSV yolu

## Ciktilar

- `outputs/academics.csv`: kisiler bazinda veri
- `outputs/boxplots.png`: unvan bazli kutu grafikler
- `outputs/bar_means.png`: unvan bazli ortalama bar grafikleri
- `outputs/report.txt`: korelasyon ve ozet istatistikler

Not: Ciktilar `outputs/` klasorune yazilir ve yeniden calistirmada uzerine
yazilabilir.

## Notlar

- YÖK Akademik sayfalari oturum temelli calisir; istekler tek bir `Session`
  ile yapilir.
- `SCI/SSCI` sayimi, makale sayfasindaki etiketlere gore hesaplanir.
