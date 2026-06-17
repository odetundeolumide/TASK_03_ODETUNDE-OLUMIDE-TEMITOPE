# Customer Segmentation via Unsupervised Learning

**DecodeLabs Project 3 — Industrial Training Kit, Batch 2026**
RFM Feature Engineering · StandardScaler · PCA · K-Means · Business Personas

---

## Author

**Odetunde Olumide Temitope**
LinkedIn: [www.linkedin.com/in/olumide-temitope-odetunde-209924201](https://www.linkedin.com/in/olumide-temitope-odetunde-209924201)

---

## Project Overview

This project takes a year of raw, unlabeled e-commerce transactions and turns them into a set of mathematically-grounded customer segments that a marketing team could act on directly. There are no labels to predict here — the task is to discover hidden structure in customer behavior using distance-based clustering, compress a wide feature space with PCA so that distance actually means something, and then translate the resulting clusters back into human-readable business personas with concrete recommended actions.

The pipeline follows a deliberate four-phase structure:

```
RAW DATA  →  CLEAN  →  FEATURE ENGINEER  →  SCALE  →  PCA  →  K-MEANS  →  PERSONAS
```

## Dataset

- **Source:** [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **File:** `online_retail_II.xlsx`, loaded locally (two sheets, "Year 2009-2010" and "Year 2010-2011", concatenated into one table)
- **Raw size:** 1,067,371 transaction line items across 8 columns (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country)
- **Coverage:** real UK e-commerce transactions spanning December 2009 to December 2011

The raw file is not bundled with the notebook — download it from the UCI link above and place it in the same folder as the notebook before running.

## Tools & Libraries

| Category | Libraries |
|---|---|
| Data handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | scikit-learn (`StandardScaler`) |
| Dimensionality reduction | scikit-learn (`PCA`) |
| Clustering | scikit-learn (`KMeans`) |
| Evaluation | scikit-learn (`silhouette_score`, `silhouette_samples`) |
| Elbow detection | `kneed` (Kneedle algorithm) |

## Project Workflow — Step by Step

### Section 1: Environment Setup & Data Loading
All required libraries are imported, a `figures/` directory is created so every chart produced later is saved as a PNG in addition to being displayed inline, and the two sheets of `online_retail_II.xlsx` are read and concatenated into a single DataFrame. The result: **1,067,371 rows × 8 columns** loaded successfully.

### Section 2: Exploratory Data Analysis (EDA)
Before any cleaning, the raw data is profiled for structure and problems. `df.info()` and a null-value count reveal 4,382 missing product descriptions and, more importantly, **243,007 missing Customer IDs** — these are anonymous or guest transactions that can't be attributed to a segmentable customer. `df.describe()` shows the numeric ranges, and histograms of `Quantity` and `Price` (clipped for readability) reveal a mix of negative quantities (returns) and a heavily right-skewed price distribution.

### Section 3: Data Cleaning & Feature Engineering

**3.1 — Cleaning.** Four categories of rows are identified as unsuitable for segmentation and removed: cancelled invoices (Invoice codes starting with "C"), rows with missing Customer ID, and rows with non-positive quantity or price (data entry errors or adjustments). After cleaning, the dataset shrinks from 1,067,371 to **805,549 rows**, leaving **5,878 unique customers** to segment.

**3.2 — RFM feature engineering.** The industry-standard RFM framework is computed per customer:
- **Recency** — days since the customer's last purchase, measured against a snapshot date set one day after the last transaction in the dataset (2011-12-10)
- **Frequency** — count of distinct invoices (orders) per customer
- **Monetary** — total revenue (Quantity × Price, summed) per customer

**3.3 — Extended behavioral features.** Beyond core RFM, nine additional features are engineered to give PCA a richer space to compress: `TotalItems`, `AvgOrderValue`, `MaxOrderValue`, `StdOrderValue` (spend variability), `UniqueProducts` (basket variety), `AvgItemsPerOrder`, `ActiveDays` (customer lifespan), `FirstPurchaseDays` (customer age/tenure), and `IsUK` (a binary flag for whether the customer's most common country is the United Kingdom). In total, **12 customer-level features** are produced for all 5,878 customers.

**3.4 — Outlier treatment (Winsorization).** K-Means relies on Euclidean distance, so a handful of extreme outliers (e.g. a single wholesale-style buyer) can distort the entire distance calculation and get isolated into their own meaningless "cluster." Rather than deleting these customers, the five most outlier-prone columns (`Monetary`, `TotalItems`, `AvgOrderValue`, `MaxOrderValue`, `StdOrderValue`) are capped at their 99th percentile — every customer stays in the dataset, but extreme values are pulled in to a realistic ceiling. This affects 59 customers per capped feature; for example, `Monetary` is capped at £29,730.42 down from an original maximum of £608,821.65.

### Section 4: Phase 1 — Standardization (SCALE)
K-Means' Euclidean distance formula treats every feature as if it's on the same scale, so a feature measured in thousands of pounds (`Monetary`) would otherwise completely dominate a binary 0/1 feature (`IsUK`). `StandardScaler` is applied to convert every feature to a z-score (`z = (x − μ) / σ`), giving each feature a mean of 0 and standard deviation of 1 — verified explicitly in the notebook output, where every feature's post-scaling mean and std come back as ≈0 and ≈1.

### Section 5: Phase 2 — Dimensionality Reduction (COMPRESS via PCA)
With 12 standardized features, the "curse of dimensionality" sets in: in high-dimensional space, points become nearly equidistant from each other and Euclidean distance loses its power to discriminate. PCA is fit across all 12 components first to inspect the full explained-variance curve, then the **95% cumulative variance rule** is applied to choose how many components to keep. In this run, just **8 principal components retain 96.4% of the total variance** — a 12D → 8D compression that discards only noise. Component loadings are also extracted to interpret what each PC represents in terms of the original behavioral features (for example, PC1 loads heavily and positively on `Monetary`, `TotalItems`, and `Frequency` — essentially a general "customer value" axis).

### Section 6: Phase 3a — Finding Optimal K
K-Means cannot determine its own number of clusters — it forces the data into whatever K is supplied, so the right K has to be found independently using two diagnostic methods that ideally agree:

- **Gatekeeper 1 — Elbow Method.** WCSS (within-cluster sum of squares) is computed for K = 2 through 10, and the Kneedle algorithm pinpoints the mathematical inflection point rather than relying on eyeballing the chart. In this run, the elbow lands at **K = 6**.
- **Gatekeeper 2 — Silhouette Score.** This metric measures how well-separated and cohesive each cluster is, ranging from −1 (wrong cluster) to +1 (perfectly separated). The score is computed for every K from 2 to 10, and the maximum is selected. In this run, the best silhouette score is at **K = 2** (score = 0.6068) — notably higher than the score at any other K, including the elbow's suggested K = 6 (0.3356).

**The two gatekeepers disagree here** (Elbow says 6, Silhouette says 2). The notebook's resolution rule favors the Silhouette Score when they conflict, since it directly measures cluster quality (cohesion + separation) rather than just diminishing returns on WCSS — so the final choice is **K = 2**. This is a worth noting as a genuine trade-off: K = 6 would likely surface more nuanced sub-segments at the cost of less statistically "clean" separation, while K = 2 gives a simpler, more robust, but coarser split.

### Section 7: Phase 3b — K-Means Clustering (CLUSTER)
The final K-Means model is fit with K = 2 on the 8-component PCA space, using `k-means++` initialization and 20 independent runs (`n_init=20`) to ensure stable centroids. The result: a **silhouette score of 0.6068** and two clusters of very different sizes — **5,490 customers (93.4%)** in Cluster 0 and **388 customers (6.6%)** in Cluster 1. The clusters are visualized as a 2D scatter plot (PC1 vs PC2), a 3D scatter plot (PC1/PC2/PC3), and a silhouette sample plot showing the per-customer fit quality within each cluster.

### Section 8: Phase 4 — Centroid Reverse-Engineering (TRANSLATE)
K-Means centroids exist in abstract PCA-space coordinates that mean nothing to a business stakeholder. To make them interpretable, each centroid is passed back through `pca_main.inverse_transform()` and then `scaler.inverse_transform()`, converting it step by step from PCA-space → scaled-space → original units. The result is a centroid expressed in real terms:

| | Recency (days) | Frequency (orders) | Monetary (£) | AvgOrderValue (£) | UniqueProducts |
|---|---|---|---|---|---|
| Cluster 0 | 210.2 | 4.5 | 1,496.0 | 23.2 | 71.4 |
| Cluster 1 | 76.1 | 32.3 | 14,478.2 | 106.3 | 232.2 |

A radar-style "cluster fingerprint" plot and boxplots of RFM values per cluster make the contrast visually clear: Cluster 1 customers buy far more recently, far more often, and spend roughly 10x more per customer than Cluster 0.

### Section 9: Business Persona Matrix
This is where the analysis becomes actionable. Rather than sorting clusters into a generic set of template categories, the persona assignment is data-driven and tied directly to this run's centroid statistics: whichever cluster has the higher mean Monetary value is labeled "Champions," and the remaining cluster is labeled "New / Dormant." Because the rule is computed from the data rather than hardcoded to a cluster index, it re-evaluates correctly if the notebook is ever re-run on different data. For this run, that resolves to:

| Cluster | Persona | Size | Recency | Frequency | Monetary | Recommended Action |
|---|---|---|---|---|---|---|
| 0 | New / Dormant | 5,490 (93.4%) | 210 days | 4.5 orders | £1,478 | Re-engagement campaigns, onboarding nurture flows, and incentives to drive a repeat purchase |
| 1 | Champions | 388 (6.6%) | 77 days | 30.9 orders | £14,727 | Reward with exclusive perks, early product access, and VIP loyalty treatment |

A pie chart shows the size split between personas, and a heatmap visualizes the normalized RFM means side by side with their raw values annotated.

### Section 10: Conclusions & Next Steps
The notebook closes with a summary table recapping the four phases (Scale → Compress → Cluster → Translate), the key formulas used throughout (z-score, Euclidean distance, the 95% PCA threshold, WCSS, silhouette score, and the inverse-transform formula for centroids), and a results export: the final customer-level table (Customer ID, Recency, Frequency, Monetary, Cluster, PersonaName) is saved to `customer_segments_output.csv` for handoff to a marketing team. Suggested next steps include automating cluster assignment for new customers, tracking cluster migration over time, A/B testing the targeted actions per persona, and trying DBSCAN or hierarchical clustering as complementary methods for non-spherical or hierarchical cluster structures.

## Interpreting the Results

The headline finding is a stark 93/7 split: the overwhelming majority of customers (93.4%) fall into a low-engagement "New/Dormant" segment, averaging just 4.5 orders and £1,478 in lifetime spend with a long 210-day gap since their last purchase, while a small "Champions" segment (6.6% of customers) drives disproportionate value — averaging £14,727 in spend, nearly 10x more than the larger group, with far more frequent and recent purchases. This is a classic retail pattern (a small share of customers generating a large share of revenue), and it directly motivates the differentiated actions in the persona matrix: a re-engagement push for the dormant majority, and a retention/loyalty push for the high-value minority. It's also worth flagging the K=2 vs K=6 trade-off discussed in Section 6 as an explicit caveat — a finer K=6 segmentation might reveal useful sub-groups within the dormant majority (e.g. genuinely new customers vs. customers actively churning), which would be a natural follow-up analysis.

## Repository Structure

```
customer-segmentation-project/
├── Project3_Customer_Segmentation.ipynb   # Main notebook (this project)
├── online_retail_II.xlsx                  # Dataset (download separately from UCI)
├── customer_segments_output.csv           # Exported output: customer-level clusters + personas
├── figures/                                # All charts saved here as PNG
│   ├── eda_raw_distributions.png
│   ├── outlier_capping_effect.png
│   ├── rfm_distributions.png
│   ├── feature_correlation_heatmap.png
│   ├── scaling_before_after.png
│   ├── pca_explained_variance.png
│   ├── pca_loadings.png
│   ├── elbow_method.png
│   ├── silhouette_scores_by_k.png
│   ├── clusters_2d_pca.png
│   ├── clusters_3d_pca.png
│   ├── silhouette_sample_plot.png
│   ├── cluster_fingerprints.png
│   ├── rfm_boxplots_by_cluster.png
│   └── persona_pie_heatmap.png
└── README.md
```

## How to Run

1. Download `online_retail_II.xlsx` from the [UCI dataset page](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it in the same folder as the notebook.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn kneed openpyxl
   ```
3. Open `Project3_Customer_Segmentation.ipynb` in Jupyter and run all cells top to bottom.

## Possible Extensions

- Re-run clustering at K = 6 (the Elbow Method's suggestion) and compare the resulting personas against the simpler K = 2 split
- Try DBSCAN or hierarchical clustering, which don't assume spherical clusters the way K-Means does
- Track how individual customers migrate between segments over time (e.g. quarter to quarter) to catch churn early
- Build a lightweight classifier that assigns new customers to a segment in real time based on their first few transactions

## Connect

If you'd like to discuss this project or other data science work, feel free to connect on LinkedIn: [www.linkedin.com/in/olumide-temitope-odetunde-209924201](https://www.linkedin.com/in/olumide-temitope-odetunde-209924201)
