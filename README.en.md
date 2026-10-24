# Hotel booking cancellation prediction

A model that estimates, at the moment a booking is made, how likely it is to be canceled. Intended use: telling the hotel which bookings are worth a call before arrival.

*[Versão em português](README.md)*

## Result

On the test period (arrivals from May to August 2017, never used to train or choose anything):

- Among the 10% of bookings the model flags as riskiest, **67% to 73% are actually canceled**. Picking at random gives 35%, and a simple segment × lead time table gives 54%.
- ROC-AUC of **0.765** (0.745 for the conservative version).
- Accuracy of **72.5%**, with 62% precision and 55% recall. Details and caveats under [Accuracy and F1](#accuracy-and-f1).
- What matters most: lead time, special requests, travel agent and market segment.

![Test result](figures/resultado_teste.png)

The result is given as a range (67% to 73%) because of one variable with a caveat, explained under [Limitations](#limitations).

## Accuracy and F1

To measure hits, the probability is turned into a yes or no with a threshold and the prediction is compared with what happened. On test:

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Guess that no booking cancels | 65.3% | – | 0% | 0 |
| Guess that every booking cancels | 34.7% | 34.7% | 100% | 0.52 |
| LightGBM, default threshold 0.5 | 72.7% | 67.9% | 40.2% | 0.51 |
| **LightGBM, profit threshold 0.31 (official)** | **72.5%** | **61.8%** | **54.5%** | **0.58** |
| LightGBM, best-F1 threshold 0.26 | 71.4% | 58.2% | 62.8% | 0.60 |

- **Accuracy alone is misleading.** 72.5% looks high, but guessing that nobody cancels already gets 65.3% right, because most bookings are not canceled. The real gain in accuracy is about 7 points.
- **Precision and recall say more:** of the bookings the model flags as risky, 62% cancel; of the cancellations that happened, it catches 55%.
- **The default 0.5 threshold is a poor choice here:** it catches only 40% of cancellations, and its F1 falls below the guess that every booking cancels. Each metric points to a different threshold; the official one was chosen by profit (see [Decision](#decision-who-to-call)), and all thresholds were chosen on validation.

![Confusion matrix on test](figures/matriz_confusao.png)

*Charts and notebooks are in Portuguese.*

## Data

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle): 119,390 bookings from two hotels in Portugal, a city hotel in Lisbon and a resort in the Algarve, with arrivals from July 2015 to August 2017. Original source: Antonio, Almeida and Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22.

## What was left out, and why

For each column the question was: does this information exist on the day of the booking? Six columns failed, because they are only filled in later (at check-in or cancellation) and would hand the answer to the model.

| Column | Why it is out |
|---|---|
| `reservation_status`, `reservation_status_date` | They are the answer itself: the final status and the date of the cancellation or check-out |
| `assigned_room_type` | The room is assigned at check-in. A room different from the one booked almost always means the guest showed up |
| `booking_changes` | Counts changes up to check-in or cancellation |
| `country` | Canceled bookings keep Portugal as the default value: in 2015, 96% of canceled bookings are marked as Portugal, against 39% of the ones that checked out |
| `required_car_parking_spaces` | Bookings that asked for parking cancel exactly 0% of the time. The parking spot is recorded on arrival |

![Parking: 0% cancellation](figures/vazamento_estacionamento.png)

Keeping the four subtler leaking columns would raise validation ROC-AUC from 0.815 to 0.875. With `reservation_status` it would reach 1.0. The number looks better, but the model would be using information that does not exist when it would actually be used.

![How much leakage inflates the result](figures/vazamento_inflacao.png)

31,980 identical rows were also removed (rooms from the same group booking, with no ID to tell them apart; if kept, a few large groups would dominate training), along with 198 records that make no sense, such as a negative daily rate or a booking with no guests.

## How it was done

**1. Time-based split, not random.** Training on arrivals from Jul/2015 to Dec/2016, validation on Jan to Apr/2017 and test on May to Aug/2017. The cancellation rate rises over time (25% in training, 28% in validation, 35% in test), and a time-based split shows how the model does on a period it has never seen.

![Cancellation by arrival month](figures/cancelamento_por_mes.png)

**2. Baselines before any model:** random choice, a rule based on lead time only, and a table with the cancellation rate by segment × lead time band.

**3. Models:** logistic regression, random forest and LightGBM, all with the same preprocessing inside a scikit-learn `Pipeline`. LightGBM was chosen on validation. Hyperparameter tuning moved ROC-AUC by less than 0.005, so the original configuration was kept.

| Model | ROC-AUC validation | ROC-AUC test | Canceled in top 10%, validation | Canceled in top 10%, test |
|---|---|---|---|---|
| Random | 0.500 | 0.500 | 28.4% | 34.7% |
| Segment × lead time table | 0.735 | 0.675 | 55.6% | 53.9% |
| Logistic regression | 0.790 | | 74.5% | |
| Random forest | 0.812 | | 72.9% | |
| LightGBM without special requests | 0.789 | 0.745 | 65.6% | 67.5% |
| **LightGBM (final model)** | **0.815** | **0.765** | **74.1%** | **72.7%** |

**4. Explanation.** SHAP values summed per variable. The directions make business sense: bookings made far in advance, with no special requests and coming from online travel agents cancel more. The `Non Refund` deposit has a strong effect because, in this dataset, it marks room blocks from groups and tour operators, which get canceled when they are not filled.

![Effect of each variable](figures/efeito_das_variaveis.png)

**5. Test:** used only once, with the model, threshold and cost scenario decided beforehand.

## Decision: who to call

The model gives a probability, but the hotel needs a yes or no. Scenario used (an assumption, to be replaced by the hotel's real numbers): contacting a booking with an incentive to confirm costs €30, and when the booking was going to be canceled the contact recovers 25% of its value (daily rate × nights).

- **On validation**, the best cut-off was 0.31: contact 31% of bookings, for a profit of €141k over four months. Contacting everyone would lose €35k.
- **On test, this fixed cut-off did not hold up:** it made €423k, less than contacting everyone (€451k). In summer, bookings are worth almost twice as much (€561 against €313 on validation) and cancel more, so contacting pays off even for low-risk bookings.
- **Lesson learned** (analysis done after seeing the test, so it is not the official result): a rule taken straight from the cost scenario, contact when *probability × 25% × booking value > €30*, wins in both periods (€166k on validation and €569k on test) and adapts to the season on its own. It is the recommendation for a next version.

## Limitations

- **Special requests** is the most important variable and may include requests made after booking, close to arrival, by guests who were going to show up anyway. The data cannot confirm this. Without it, the test result drops from 73% to 67% in the top 10%, which is why the result is given as a range.
- The model gets the order right (who is riskier) but underestimates the level of risk when the cancellation rate rises: it predicted 28.8% on average on test, and the actual rate was 34.7%. In real use it would need periodic retraining.
- Two hotels in Portugal, from 2015 to 2017. For another hotel, the model would have to be retrained.
- The cost scenario is assumed. The dataset does not state the currency; values were treated as euros.

## How to run

Python 3.11. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

Or with pip: `python -m venv .venv` and `pip install -r requirements.txt`.

Download `hotel_bookings.csv` from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand), put it in `data/raw/` and run the notebooks in order:

| Notebook | What it does |
|---|---|
| `notebooks/01_dados_eda.ipynb` | Cleaning, duplicates, leakage audit and exploratory analysis. Saves the clean dataset |
| `notebooks/02_separacao.ipynb` | Splits training, validation and test by date |
| `notebooks/03_modelos.ipynb` | Baselines, models and model choice. Measures how much leakage would inflate the result |
| `notebooks/04_decisao_e_teste.ipynb` | Cost-based threshold, SHAP explanation, test result and accuracy metrics (accuracy, precision, recall, F1) |

`notebooks/graficos.py` and `notebooks/avaliacao.py` hold the chart functions and metrics used by the notebooks.

## Author

Caio Goia · [LinkedIn](https://www.linkedin.com/in/caio-goia) · [GitHub](https://github.com/caiogoia123)
