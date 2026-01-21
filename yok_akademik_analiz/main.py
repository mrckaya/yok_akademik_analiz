import argparse
import os
import re
import time
import unicodedata
import urllib3
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

BASE_URL = "https://akademik.yok.gov.tr/AkademikArama/"
SEARCH_ENDPOINT = urljoin(BASE_URL, "AkademisyenArama")
ROOT_URL = "https://akademik.yok.gov.tr/"

TITLE_ORDER = {
    "Doktor Öğretim Üyesi": 1,
    "Doçent": 2,
    "Profesör": 3,
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text.strip().upper()


def extract_title(info_text: str) -> Optional[str]:
    if not info_text:
        return None
    normalized = normalize_text(info_text)
    normalized = re.sub(r"[^A-Z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if re.search(r"\bPROFESOR\b|\bPROF\b", normalized):
        return "Profesör"
    if re.search(r"\bDOCENT\b|\bDOC\b|\bDOÇENT\b|\bDOÇ\b", normalized):
        return "Doçent"
    if re.search(r"\bDOKTOR\b.*\bOGRETIM\b.*\bUYE", normalized):
        return "Doktor Öğretim Üyesi"
    if re.search(r"\bDOKTOR\b.*\bOGR\b.*\bUYE", normalized):
        return "Doktor Öğretim Üyesi"
    if re.search(r"\bDR\b.*\bOGRETIM\b.*\bUYE", normalized):
        return "Doktor Öğretim Üyesi"
    if re.search(r"\bDR\b.*\bOGR\b.*\bUYE", normalized):
        return "Doktor Öğretim Üyesi"
    return None


def extract_ana_bilim_dali(info_text: str) -> Optional[str]:
    if not info_text:
        return None
    match = re.search(
        r"([^/]*?)\s*ANAB[İI]L[İI]M\s*DALI", info_text, flags=re.IGNORECASE
    )
    if match:
        return re.sub(
            r"\s*ANAB[İI]L[İI]M\s*DALI",
            "",
            match.group(1),
            flags=re.IGNORECASE,
        ).strip()
    return None


def extract_specializations(info_text: str) -> List[str]:
    if not info_text:
        return []
    match = re.search(r"Temel\s+Alan[ıi]\s*(.*)", info_text, flags=re.IGNORECASE)
    if not match:
        return []
    tail = match.group(1)
    tail = re.sub(r"\b\S+\[at\]\S+\b", "", tail)
    tail = re.sub(r"\b\S+@\S+\b", "", tail)
    tail = tail.replace("/", " ")
    tail = re.sub(r"\s+", " ", tail).strip()
    parts = [p.strip() for p in tail.split(",") if p.strip()]
    refined: List[str] = []
    for part in parts:
        if "Mühendisliği" in part and len(part.split()) > 4:
            split_idx = part.find("Mühendisliği")
            left = part[: split_idx + len("Mühendisliği")].strip()
            right = part[split_idx + len("Mühendisliği") :].strip()
            if left:
                refined.append(left)
            if right:
                refined.append(right)
        else:
            refined.append(part)
    refined = [item for item in refined if item]
    return refined


def parse_specializations_from_view(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    header = soup.select_one("h4") or soup.select_one("h3") or soup.select_one("h5")
    text = header.get_text(" ", strip=True) if header else ""
    if not text:
        return []
    return extract_specializations(text)


def dedupe_preserve_order(values: List[str]) -> List[str]:
    seen_norm = set()
    deduped: List[str] = []
    for value in values:
        if not value:
            continue
        norm = normalize_text(value)
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        deduped.append(value)
    return deduped


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Referer": BASE_URL,
        }
    )
    return session


def fetch_html(
    session: requests.Session, url: str, *, method: str = "GET", verify_ssl: bool = True, data=None
) -> str:
    response = session.request(method, url, data=data, timeout=30, verify=verify_ssl)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def fetch_with_retry(
    session: requests.Session,
    url: str,
    *,
    method: str = "GET",
    verify_ssl: bool,
    data=None,
    retries: int = 2,
    backoff: float = 0.6,
) -> Optional[str]:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return fetch_html(
                session, url, method=method, verify_ssl=verify_ssl, data=data
            )
        except requests.HTTPError as exc:
            last_error = exc
            time.sleep(backoff + attempt * backoff)
    if last_error:
        return None
    return None


def parse_search_results(html: str) -> List[Dict[str, Optional[str]]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr[id^=authorInfo_]")
    results = []
    for row in rows:
        tds = row.find_all("td")
        profile_link = None
        if row.find("a", href=True):
            profile_link = row.find("a", href=True)["href"]
        image = row.find("img")
        name = image.get("alt", "").strip() if image else None
        info_text = tds[2].get_text(" ", strip=True) if len(tds) > 2 else ""
        email = tds[3].get_text(" ", strip=True) if len(tds) > 3 else None
        author_id = tds[4].get_text(" ", strip=True) if len(tds) > 4 else None

        results.append(
            {
                "name": name,
                "info_text": info_text,
                "unvan": extract_title(info_text),
                "ana_bilim_dali": extract_ana_bilim_dali(info_text),
                "email": email,
                "author_id": author_id,
                "profile_url": profile_link,
            }
        )
    return results


def get_pagination_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a_tag in soup.select("a[href]"):
        href = a_tag.get("href") or ""
        if (
            "AramaFiltrele" in href
            or "viewAuthorArticle" in href
            or "viewAuthorBook" in href
        ):
            if href and href not in links:
                links.append(href)
    return links


def normalize_url(link: str) -> str:
    if link.startswith("http://") or link.startswith("https://"):
        return link
    if link.startswith("/"):
        return urljoin(ROOT_URL, link.lstrip("/"))
    if link.startswith("AkademikArama/"):
        return urljoin(ROOT_URL, link)
    return urljoin(BASE_URL, link)


def search_academics(
    session: requests.Session,
    search_terms: List[str],
    *,
    verify_ssl: bool,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Optional[str]]]:
    results: List[Dict[str, Optional[str]]] = []
    for term in search_terms:
        payload = {
            "aramaTerim": term,
            "yazarCheckbox": "on",
            "islem": "1",
        }
        first_html = fetch_with_retry(
            session, SEARCH_ENDPOINT, method="POST", verify_ssl=verify_ssl, data=payload
        )
        if not first_html:
            continue
        results.extend(parse_search_results(first_html))

        pagination_links = get_pagination_links(first_html)
        for idx, link in enumerate(pagination_links, start=2):
            if max_pages and idx > max_pages:
                break
            page_html = fetch_with_retry(
                session, normalize_url(link), verify_ssl=verify_ssl
            )
            if not page_html:
                continue
            results.extend(parse_search_results(page_html))

    unique: List[Dict[str, Optional[str]]] = []
    seen = set()
    for item in results:
        key = item.get("author_id") or f"{item.get('name')}|{item.get('email')}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def parse_profile_page(html: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    book_link = None
    article_link = None
    book_anchor = soup.select_one("li#booksMenu a[href]")
    article_anchor = soup.select_one("li#articleMenu a[href]")
    if book_anchor:
        book_link = book_anchor["href"]
    if article_anchor:
        article_link = article_anchor["href"]

    labels = [span.get_text(strip=True) for span in soup.select("span.label")]
    labels = [label for label in labels if not label.lower().startswith("son")]
    return book_link, article_link, labels


def parse_publications(html: str) -> List[Dict[str, List[str]]]:
    soup = BeautifulSoup(html, "html.parser")
    publications = []
    for row in soup.select("table.table tr"):
        title_tag = row.select_one("span.baslika")
        if not title_tag:
            continue
        title = title_tag.get_text(" ", strip=True)
        labels = [label.get_text(strip=True) for label in row.select("span.label")]
        publications.append({"title": title, "labels": labels})
    if publications:
        return publications
    for strong in soup.select("strong"):
        text = strong.get_text(" ", strip=True)
        if not re.match(r"^\d+\.\s+", text):
            continue
        title = re.sub(r"^\d+\.\s*", "", text).strip()
        if title:
            publications.append({"title": title, "labels": []})
    return publications


def is_sci_label(label: str) -> bool:
    label_upper = label.upper()
    return label_upper == "SCI" or label_upper == "SSCI"


def get_publication_counts(
    session: requests.Session,
    url: Optional[str],
    *,
    verify_ssl: bool,
    dedupe: bool = True,
) -> Tuple[int, int]:
    if not url:
        return 0, 0
    html = fetch_with_retry(
        session, normalize_url(url), verify_ssl=verify_ssl, retries=2
    )
    if not html:
        return 0, 0
    publications = parse_publications(html)
    page_links = get_pagination_links(html)
    for link in page_links:
        page_html = fetch_with_retry(
            session, normalize_url(link), verify_ssl=verify_ssl, retries=1
        )
        if not page_html:
            continue
        publications.extend(parse_publications(page_html))
    if dedupe:
        unique: Dict[str, Dict[str, List[str]]] = {}
        for pub in publications:
            key = normalize_text(pub["title"])
            if key not in unique:
                unique[key] = pub
        total = len(unique)
        sci_count = sum(
            1
            for pub in unique.values()
            if any(is_sci_label(label) for label in pub.get("labels", []))
        )
    else:
        total = len(publications)
        sci_count = sum(
            1
            for pub in publications
            if any(is_sci_label(label) for label in pub.get("labels", []))
        )
    return total, sci_count


def filter_by_department(
    academics: List[Dict[str, Optional[str]]], target_department: str
) -> List[Dict[str, Optional[str]]]:
    target_norm = normalize_text(target_department)
    filtered = []
    for item in academics:
        ana_bilim_dali = item.get("ana_bilim_dali") or ""
        info_text = item.get("info_text") or ""
        if target_norm in normalize_text(ana_bilim_dali) or target_norm in normalize_text(
            info_text
        ):
            filtered.append(item)
    return filtered


def plot_metrics(df: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    metrics = [
        ("toplam_makale", "Toplam Makale"),
        ("sci_ssci", "SCI/SSCI Makale"),
        ("kitap_sayisi", "Kitap Sayisi"),
    ]
    plt.figure(figsize=(14, 4))
    for idx, (col, title) in enumerate(metrics, start=1):
        plt.subplot(1, 3, idx)
        sns.boxplot(data=df, x="unvan", y=col)
        plt.title(title)
        plt.xlabel("")
        plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "boxplots.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    df_means = (
        df.groupby("unvan")[["toplam_makale", "sci_ssci", "kitap_sayisi"]]
        .mean()
        .reset_index()
    )
    df_means = df_means.melt(id_vars="unvan", var_name="metric", value_name="mean")
    sns.barplot(data=df_means, x="unvan", y="mean", hue="metric")
    plt.title("Unvanlara Gore Ortalama Uretkenlik")
    plt.xlabel("")
    plt.ylabel("Ortalama")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bar_means.png"), dpi=200)
    plt.close()


def analyze_correlation(df: pd.DataFrame) -> Tuple[float, float]:
    df = df.copy()
    df["unvan_ord"] = df["unvan"].map(TITLE_ORDER)
    df = df.dropna(subset=["unvan_ord", "toplam_makale"])
    corr, p_value = spearmanr(df["unvan_ord"], df["toplam_makale"])
    return float(corr), float(p_value)


def save_report(outdir: str, corr: float, p_value: float, df: pd.DataFrame) -> None:
    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, "report.txt")
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("Spearman korelasyonu (unvan_ord vs toplam_makale):\n")
        file_handle.write(f"Rho: {corr:.4f}\n")
        file_handle.write(f"P-degeri: {p_value:.4f}\n\n")
        file_handle.write("Unvanlara gore temel istatistikler:\n")
        file_handle.write(
            df.groupby("unvan")[["toplam_makale", "sci_ssci", "kitap_sayisi"]]
            .describe()
            .to_string()
        )
        file_handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOK Akademik verisi cekme ve analiz"
    )
    parser.add_argument(
        "--department",
        default="Bilgisayar Mühendisliği",
        help="Ana bilim dali hedefi",
    )
    parser.add_argument(
        "--search-term",
        default=None,
        help="Arama terimi (varsayilan: bolum adinin ilk kelimesi)",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-academics", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=1.5)
    parser.add_argument(
        "--insecure", action="store_true", help="SSL dogrulamayi kapat"
    )
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--data-path", default="outputs/academics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_term = args.search_term or args.department.split()[0]
    search_terms = [
        search_term,
        f"Doçent {search_term}",
        f"Docent {search_term}",
        f"Dr Öğr {search_term}",
        f"Dr Ogr {search_term}",
        f"Doktor Öğretim Üyesi {search_term}",
        f"Doktor Ogretim Uyesi {search_term}",
    ]
    search_terms = [term.strip() for term in search_terms if term.strip()]
    search_terms = list(dict.fromkeys(search_terms))
    session = build_session()
    verify_ssl = not args.insecure
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print("Analiz baslatildi, veriler hazirlaniyor...")
    academics = search_academics(
        session, search_terms, verify_ssl=verify_ssl, max_pages=args.max_pages
    )

    
    filtered = filter_by_department(academics, args.department)
    records = []
    for idx, academic in enumerate(filtered, start=1):
        if args.max_academics and idx > args.max_academics:
            break
        profile_url = academic.get("profile_url")
        if not profile_url:
            continue
        profile_html = fetch_with_retry(
            session, normalize_url(profile_url), verify_ssl=verify_ssl, retries=2
        )
        if not profile_html:
            continue
        book_link, article_link, labels = parse_profile_page(profile_html)
        view_specializations: List[str] = []
        view_html = fetch_with_retry(
            session, normalize_url("view/viewAuthor.jsp"), verify_ssl=verify_ssl, retries=2
        )
        if view_html:
            view_specializations = parse_specializations_from_view(view_html)
        article_count, sci_count = get_publication_counts(
            session, article_link, verify_ssl=verify_ssl
        )
        book_count, _ = get_publication_counts(
            session, book_link, verify_ssl=verify_ssl, dedupe=False
        )

        merged_specializations = dedupe_preserve_order(
            extract_specializations(academic.get("info_text") or "")
            + view_specializations
            + labels
        )

        records.append(
            {
                "name": academic.get("name"),
                "unvan": academic.get("unvan"),
                "ana_bilim_dali": academic.get("ana_bilim_dali"),
                "uzmanlik_alani": " | ".join(merged_specializations)
                if merged_specializations
                else None,
                "toplam_makale": article_count,
                "sci_ssci": sci_count,
                "kitap_sayisi": book_count,
            }
        )
        time.sleep(args.sleep)

    df = pd.DataFrame(records)
    df = df[df["unvan"].isin(TITLE_ORDER.keys())]
    os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
    df.to_csv(args.data_path, index=False, encoding="utf-8-sig", sep=";")

    if df.empty:
        print("Filtrelenen akademisyen bulunamadi.")
        return

    plot_metrics(df, args.outdir)
    corr, p_value = analyze_correlation(df)
    save_report(args.outdir, corr, p_value, df)

    print(f"Kayit sayisi: {len(df)}")
    print(f"Rho: {corr:.4f}  P-degeri: {p_value:.4f}")
    print(f"CSV: {args.data_path}")
    print(f"Grafikler: {args.outdir}")


if __name__ == "__main__":
    main()
