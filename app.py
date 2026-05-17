
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DAP–Analytic Avengers",
    page_icon="dap.png",
    layout="wide",
)
st.markdown("""
        <style>
        /* =========================================
           DYNAMIC THEME VARIABLES
           ========================================= */
        :root {
            --text-main: #0f172a;       /* Dark slate for high contrast */
            --text-sub: #64748b;        /* Medium slate for reading */
            --card-bg: #ffffff;         /* Clean white */
            --card-border: #e2e8f0;     /* Light gray border */
            --shadow-color: rgba(15, 23, 42, 0.08);
            --btn-bg: #ffffff;
            --btn-border: #cbd5e1;
            --accent-gradient: linear-gradient(135deg, #6366f1, #0ea5e9); /* Indigo to Sky */
            --btn-hover-bg: #f8fafc;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --text-main: #f8fafc;       /* Off-white for dark mode */
                --text-sub: #94a3b8;        /* Light slate */
                --card-bg: #1e293b;         /* Deep slate background */
                --card-border: #334155;     /* Dark border */
                --shadow-color: rgba(0, 0, 0, 0.4);
                --btn-bg: #0f172a;
                --btn-border: #334155;
                --accent-gradient: linear-gradient(135deg, #818cf8, #38bdf8); /* Brighter Indigo to Sky */
                --btn-hover-bg: #1e293b;
            }
        }

        /* =========================================
           COMPONENT STYLING
           ========================================= */
        /* Gradient Title */
        .main-header {
            font-size: 2.4rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            padding-top: 1rem;
        }

        /* Refined Subtext */
        .home-subtext {
            font-size: 1.1rem;
            color: var(--text-sub);
            margin-bottom: 2.5rem;
            font-weight: 400;
            letter-spacing: 0.3px;
        }

        /* Modern Information Card */
        .guide-card {
            background-color: #f5f9ff;
            padding: 1.8rem;
            border-radius: 20px;
            border: 1px solid #4a90e2;
            box-shadow: 0 10px 30px var(--shadow-color);
            color: #1f2937;
            position: relative;
            overflow: hidden;
        }

        .guide-card h4 {
            color: #1f2937 !important;
        }

        .guide-card p {
            color: #374151 !important;
        }

        .guide-card b {
            color: #111827 !important;
        }

        /* Premium Button Overrides */
        div.stButton > button {
            border-radius: 16px;
            padding: 1.2rem 1rem;
            font-weight: 600;
            font-size: 1.05rem;
            min-height: 110px;
            background-color: var(--btn-bg);
            color: var(--text-main);
            border: 1px solid var(--btn-border);
            box-shadow: 0 4px 14px var(--shadow-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Button Hover State */
        div.stButton > button:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px var(--shadow-color);
            border-color: #6366f1; /* Accent color border on hover */
            background-color: var(--btn-hover-bg);
        }

        /* Button Active/Click State */
        div.stButton > button:active {
            transform: translateY(0px);
            box-shadow: 0 4px 10px var(--shadow-color);
        }
        </style>
        """, unsafe_allow_html=True)

col_left, col_right = st.columns([4, 1])
with col_left:
    st.markdown('<div class="main-header">DAP by Analytic Avengers 🐾</div>', unsafe_allow_html=True)
    st.caption("Predictive health analysis powered by the Dog Aging Project dataset. A Journey in Canine Wellness!")
with col_right:
    st.image("dap.png",width=200)
# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        # navigation
        "page":             "home",   # "home" | "cslb" | "regression"
        "breed":            None,
        # cslb first-pass result (persisted across reruns)
        "cslb_predicted":   None,
        "cslb_inputs":      None,
        # 6-month follow-up
        "fu_submitted":     False,
        "fu_pace":     0, "fu_stare":     0, "fu_defecate": 0,
        "fu_food":     0, "fu_recognize": 0, "fu_active":   0,
        # regression results
        "reg_results":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading the 'Dog Aging Project' dataset…")
def load_data():
    df = pd.read_csv("final.csv")
    data = df.set_index("dog_id")
    data.replace(np.nan, 0, inplace=True)
    breed = df["Breed"].unique().tolist()
    return data,breed


@st.cache_resource(show_spinner="Training CSLB prediction model…")
def train_cslb_model():
    data,breed = load_data()
    final = data
    Y = final["cslb_score"]
    X = final[[
        "dd_age_years", "dd_weight_lbs", "pa_activity_level",
        "hs_health_conditions_eye", "hs_health_conditions_ear",
        "cslb_pace", "cslb_stare", "cslb_stuck", "cslb_recognize",
        "cslb_walk_walls", "cslb_avoid", "cslb_find_food",
    ]]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

# findig the most common disease for a breed

def get_most_common_disease_for_breed(data, breed_name, close_threshold=5.0):
    DISEASE_LABEL_MAP = {
        "cancer": "Cancer",
        "gastrointestinal": "Gastrointestinal disease",
        "skin": "Skin disease",
        "oral": "Oral disease",
        "neurological": "Neurological disease",
        "kidney": "Kidney disease",
        "liver": "Liver disease",
        "cardiac": "Cardiac disease",
        "orthopedic": "Orthopedic disease",
    }
    breed_df = data[data["Breed"] == breed_name].copy()

    if breed_df.empty:
        return "No disease summary is available for this breed."

    disease_percentages = {}

    for disease_key, disease_col in DISEASE_COL_MAP.items():
        if disease_col not in breed_df.columns:
            continue

        vals = pd.to_numeric(breed_df[disease_col], errors="coerce").fillna(0)

        valid_mask = ~vals.isin([1, 3])
        vals = vals[valid_mask]

        if len(vals) == 0:
            continue

        binary_vals = vals.map(lambda x: 0 if x == 0 else 1)
        pct = binary_vals.mean() * 100
        disease_percentages[disease_key] = pct

    if not disease_percentages:
        return "No disease summary is available for this breed."

    sorted_diseases = sorted(disease_percentages.items(), key=lambda x: x[1], reverse=True)

    top1_key, top1_pct = sorted_diseases[0]

    if len(sorted_diseases) == 1:
        return f"Most common recorded disease for **{breed_name}**: **{DISEASE_LABEL_MAP[top1_key]}** ({top1_pct:.1f}%)."

    top2_key, top2_pct = sorted_diseases[1]

    if abs(top1_pct - top2_pct) <= close_threshold:
        return (
            f"Most common recorded diseases for **{breed_name}**: "
            f"**{DISEASE_LABEL_MAP[top1_key]}** ({top1_pct:.1f}%) and "
            f"**{DISEASE_LABEL_MAP[top2_key]}** ({top2_pct:.1f}%)."
        )

    return f"Most common recorded disease for **{breed_name}**: **{DISEASE_LABEL_MAP[top1_key]}** ({top1_pct:.1f}%)."

# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_COL_MAP = {
    "cancer":           "hs_health_conditions_cancer",
    "gastrointestinal": "hs_health_conditions_gastrointestinal",
    "skin":             "hs_health_conditions_skin",
    "oral":             "hs_health_conditions_oral",
    "neurological":     "hs_health_conditions_neurological",
    "kidney":           "hs_health_conditions_kidney",
    "liver":            "hs_health_conditions_liver",
    "cardiac":          "hs_health_conditions_cardiac",
    "orthopedic":       "hs_health_conditions_orthopedic",
}


def _run_logit(disease_col, variable_col, data, disease_name, hypo, psugg, nsugg):
    try:
        arr1 = data[variable_col].values
        arr2 = data[disease_col].values

        # Need at least two distinct outcome classes to run logistic regression
        if len(np.unique(arr2)) < 2 or len(np.unique(arr1)) < 2:
            return None, None

        data_reg = pd.DataFrame({"exposure_group": arr1, "outcome": arr2})

        # Drop rows where either value is NaN/inf
        data_reg = data_reg.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data_reg) < 10:
            return None, None

        exog = sm.add_constant(data_reg["exposure_group"], has_constant="add")

        try:
            result = sm.Logit(data_reg["outcome"], exog).fit(
                disp=0, warn_convergence=False, maxiter=200
            )
        except Exception:
            return None, None


        if "exposure_group" not in result.pvalues.index:
            return None, None
        if result.pvalues["exposure_group"] >= 0.05:
            return None, None


        coef = result.params["exposure_group"]
        odds_ratio = np.exp(coef)

        if not np.isfinite(odds_ratio):
            return None, None

        if odds_ratio > 1:
            finding = f"If {hypo} → odds of {disease_name} diseases are **{(odds_ratio - 1)*100:.2f}% MORE**"
            return finding, nsugg
        else:
            finding = f"If {hypo} → odds of {disease_name} diseases are **{(1 - odds_ratio)*100:.2f}% LESS**"
            return finding, psugg

    except Exception:
        return None, None


def run_regression(disease_key, variable_group, data):
    disease_col  = DISEASE_COL_MAP[disease_key]
    disease_name = disease_key
    mask   = (data[disease_col] != 1) & (data[disease_col] != 3)
    df_dis = data[mask].copy()
    df_dis[disease_col] = df_dis[disease_col].map(lambda x: 0 if x == 0 else 1)
    suggestions, findings = [], []

    if variable_group == "diet":
        specs = {
            "df_diet_consistency":                  (lambda s: s.map(lambda x: 0 if x >= 3 else 1),       "dog has a very consistent diet",                                       "Make sure your dog follows a very consistent diet!",                                           "Make sure dog's diet is not consistent!"),
            "df_appetite":                          (lambda s: s.map(lambda x: 0 if x == 1 else 1),       "dog shows good appetite",                                              "Keep an eye on your dog's appetite and make sure it is good.",                                 "Keep an eye on your dog's appetite; being always hungry is bad."),
            "df_primary_diet_component_organic":    (lambda s: s.map(lambda x: 0 if x == False else 1),   "dog's primary diet is organic",                                        "Try to feed organic foods to your dog!",                                                       "Try not to feed organic foods to your dog!"),
            "df_primary_diet_component_grain_free": (lambda s: s.map(lambda x: 0 if x == False else 1),   "dog's diet is grain free",                                             "Try to feed grain free food to your dog!",                                                     "Try not to feed grain free food to your dog!"),
            "df_primary_diet_component_change_recent":(lambda s: s.map(lambda x: 1 if x == True else 0),  "recent changes have been made to the dog's diet",                      "Try to change dog's primary diet component frequently!",                                       "Try not to change your dog's primary diet component frequently!"),
            "df_weight_change_last_year":           (lambda s: s.map(lambda x: 0 if x == 0 else 1),       "dog weight varied from last year",                                     "Keep an eye on your dog's weight! Go to vet if it changes over the year!",                     "Keep an eye on your dog's weight! Go to vet if it does not change over the year!"),
            "df_treats_frequency":                  (lambda s: s.map(lambda x: 0 if x == 0 or x == 4 else 1),"owner does not give treats at least once a day",                   "Try to give your dog treats moderately!",                                                      "Try not to give your dog treats moderately!"),
            "df_infrequent_supplements":            (lambda s: s.map(lambda x: 0 if x == False else 1),   "supplements are given less often than every day",                       "Try to give your dog supplements less often than every day!",                                   "Try to give your dog supplements every day!"),
        }
    elif variable_group == "physical_activity":
        specs = {
            "pa_activity_level":                       (lambda s: s.map(lambda x: 0 if x == 1 else 1),    "dog's lifestyle over the past year has been active",                   "Try to make sure your dog has an active lifestyle!",                                           "Try to make sure your dog does not have an active lifestyle."),
            "pa_physical_games_frequency":             (lambda s: s.map(lambda x: 1 if x <= 3 else 0),    "dog fetches items or plays games involving physical activity",          "Play games such as Frisbee with your dog or ask your dog to fetch items more.",                 "Don't play games such as Frisbee with your dog that often."),
            "pa_avg_activity_intensity":               (lambda s: s.map(lambda x: 0 if x == 1 else 1),    "average activity intensity included jogging and sprinting",             "Try to make sure that your dog does a good amount of jogging and sprinting!",                   "Try to make sure that your dog does not do too much jogging and sprinting!"),
            "pa_swim":                                 (lambda s: s.map(lambda x: 1 if x == True else 0), "dog goes swimming",                                                    "Take your dog swimming more often!",                                                           "Don't take your dog swimming too much!"),
            "pa_moderate_weather_sun_exposure_level":  (lambda s: s.map(lambda x: 1 if x <= 2 else 0),    "dog has good sun exposure on moderate weather days",                   "On moderate days (40–85°F) make sure your dog has exposure to sun.",                           "On moderate days (40–85°F) make sure your dog is in the shade."),
            "pa_moderate_weather_daily_hours_outside": (lambda s: s.map(lambda x: 0 if x in [1, 5] else 1),"dog spends less than 3 hours outdoors on moderate weather days",      "On moderate days make sure your dog spends less than 3 hours outdoors.",                       "On moderate days make sure your dog spends more than 3 hours outdoors."),
            "pa_other_aerobic_activity_frequency":     (lambda s: s.map(lambda x: 1 if x >= 3 else 0),    "dog gets other aerobic activity more than once a week",                "Do more aerobic activities that elevate heart rate more than once a week.",                     "Do aerobic activities that elevate heart rate less than once a week."),
            "pa_on_leash_walk_frequency":              (lambda s: s.map(lambda x: 1 if x >= 3 else 0),    "dog is active on a lead/leash more than once a week",                  "Your dog should be active on a lead/leash more than once a week.",                             "Your dog should be active on a lead/leash less than once a week."),
        }
    elif variable_group == "behavior":
        specs = {
            "db_aggression_level_food_taken_away":          (lambda s: s.map(lambda x: 0 if x >= 2 else 1), "dog shows No/Rarely aggression when food is taken away by a family member", "Your dog has a lesser chance of disease if it is rarely aggressive when food is taken away.", "Your dog may have a higher chance of disease if it is very aggressive when food is taken away."),
            "db_fear_level_bathed_at_home":                 (lambda s: s.map(lambda x: 0 if x >= 2 else 1), "dog shows no fear while bathed at home",                                    "Be friendly and gentle while bathing your dog so it does not become afraid.",                 "Try not to be overly friendly while bathing so the dog develops some caution."),
            "db_fear_level_nails_clipped_at_home":          (lambda s: s.map(lambda x: 0 if x >= 3 else 1), "dog shows No/Rare fear/anxiety while getting nails clipped at home",        "Be gentle while clipping nails so your dog shows no fear/anxiety.",                          "It is not so important to be extra gentle while clipping nails."),
            "db_left_alone_restlessness_frequency":         (lambda s: s.map(lambda x: 0 if x >= 3 else 1), "dog shows less/zero restlessness or agitation when left alone",             "Get help and take care of your dog if it shows fear or agitation when left alone.",           "There is nothing to fear if your dog shows agitation when left alone."),
            "db_urinates_alone_frequency":                  (lambda s: s.map(lambda x: 0 if x >= 2 else 1), "dog does not urinate when left alone",                                      "Get help and take care of your dog if it urinates when left alone.",                          "There is nothing to fear if your dog urinates while left alone."),
            "db_urinates_in_home_frequency":                (lambda s: s.map(lambda x: 0 if x >= 2 else 1), "dog does not urinate in home against objects",                              "Consult a vet if your dog is urinating in home against objects.",                             "There is nothing to worry about if your dog urinates in home against objects."),
            "db_aggression_level_unknown_aggressive_dog":   (lambda s: s.map(lambda x: 0 if x >= 2 else 1), "dog shows less/no aggression when approached by an unfamiliar dog",         "Teach your dog to be calm when approached by an unfamiliar dog.",                             "There is no risk if your dog shows aggression toward unfamiliar dogs."),
            "db_hyperactive_frequency":                     (lambda s: s.map(lambda x: 1 if x <= 2 else 0), "dog is less hyperactive/restless",                                          "Make sure your dog is not hyperactive, restless, or having trouble settling down.",           "Make sure your dog is hyperactive and restless."),
        }
    elif variable_group == "environment":
        specs = {
            "de_lifetime_residence_count":                    (lambda s: s.map(lambda x: 1 if x <= 3 else 0),    "dog's owner has had fewer than three residences in the dog's lifetime",    "Try not to change residence very frequently during your dog's lifetime.",                      "Try to move to more than three residences."),
            "de_room_or_window_air_conditioning_present":     (lambda s: s.map(lambda x: 0 if x == 0 else 1),    "owner's residence has sufficient air conditioning",                         "Ensure your residence where you live with your dog has good air conditioning.",                "Ensure your residence does not have air conditioning."),
            "de_drinking_water_is_filtered":                  (lambda s: s.map(lambda x: 0 if x == 0 else 1),    "drinking water in owner's residence is not filtered",                       "Filtered water is not a must to ensure your dog's health.",                                   "Ensure your dog has access to filtered drinking water."),
            "de_asbestos_present":                            (lambda s: s.map(lambda x: 0 if x == 0 else 1),    "asbestos is present in owner's residence floor",                            "Try to make sure that your residence floor is asbestos-free.",                                "Try to make sure there is asbestos in your residence floor."),
            "de_floor_types_wood":                            (lambda s: s.map(lambda x: 1 if x == True else 0), "dog owner's residence has a wooden floor",                                  "Try not to have wooden floors in your residence.",                                            "Try to have wooden floors in your residence."),
            "de_routine_toys":                                (lambda s: s.map(lambda x: 1 if x == True else 0), "dog regularly licks, chews, or plays with toys",                            "Ensure your dog regularly licks, chews, and plays with toys.",                                "Try not to let your pet play with chewing/licking toys regularly."),
            "de_neighborhood_has_sidewalks":                  (lambda s: s.map(lambda x: 0 if x > 0 else 1),     "neighbourhood does not have many sidewalks",                                "Try to make sure your neighbourhood does not have many sidewalks.",                           "Having too many sidewalks in a neighbourhood is not a problem."),
            "de_neighborhood_has_parks":                      (lambda s: s.map(lambda x: 1 if x == True else 0), "there are parks or green spaces within half a mile of the owner's home",   "Try to live in a neighbourhood with parks or green spaces within half a mile.",               "It is not important to have parks or green spaces nearby."),
            "de_dogpark":                                     (lambda s: s.map(lambda x: 1 if x == True else 0), "neighbourhood has dog parks",                                               "Try to live in a neighbourhood that has dog parks.",                                          "It is not necessary to have dog parks in the neighbourhood."),
            "de_recreational_spaces":                         (lambda s: s.map(lambda x: 0 if x == False else 1),"there are recreational spaces in the neighbourhood",                        "Try to choose a neighbourhood that has recreational spaces.",                                 "Try not to choose a neighbourhood that has recreational spaces."),
            "de_sitter_or_daycare":                           (lambda s: s.map(lambda x: 0 if x == False else 1),"dog has been taken to a daycare centre",                                    "Try to take your pet to a daycare centre.",                                                   "It is not necessary to take your pet to a daycare centre."),
            "de_traffic_noise_in_home_frequency":             (lambda s: s.map(lambda x: 1 if x > 1 else 0),     "traffic noise in the home is more frequent",                                "It is not an issue if there is frequent traffic noise in the neighbourhood.",                  "Try to avoid neighbourhoods where traffic noise is more frequent."),
        }
    else:
        specs = {}

    for col, (transform, hypo, psugg, nsugg) in specs.items():
        if col not in df_dis.columns:
            continue
        df_dis[col] = transform(df_dis[col])
        finding, sugg = _run_logit(disease_col, col, df_dis, disease_name, hypo, psugg, nsugg)
        if finding:
            findings.append(finding)
            suggestions.append(sugg)

    return suggestions, findings


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    hles_data,breed_list = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    breed_list  = []


if not data_loaded:
    st.error(
        "⚠️ Dataset CSV files not found.\n\n"
        "Make sure `DAP_2021_HLES_dog_owner_v1.0.csv` and "
        "`DAP_2021_CSLB_v1.0.csv` are in the same folder as `app.py`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def go_home():
    st.session_state.page           = "home"
    st.session_state.cslb_predicted = None
    st.session_state.cslb_inputs    = None
    st.session_state.fu_submitted   = False
    st.session_state.reg_results    = None

def go_cslb():
    st.session_state.page           = "cslb"
    st.session_state.cslb_predicted = None
    st.session_state.fu_submitted   = False

def go_regression():
    st.session_state.page        = "regression"
    st.session_state.reg_results = None


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    # Layout setup
    col_form, gap, col_info = st.columns([3.5, 1, 2])

    with col_form:
        st.markdown("#### 🔍 Choose an analysis")
        st.write("")  # Small spacer

        task_col1, task_col2 = st.columns(2)

        with task_col1:
            if st.button(
                    "🧠 Cognitive Dysfunction\nPrediction",
                    use_container_width=True
            ):
                go_cslb()
                st.rerun()

        with task_col2:
            if st.button(
                    "🔬 Disease Regression\nAnalysis",
                    use_container_width=True
            ):
                go_regression()
                st.rerun()

    with col_info:
        st.markdown("""
                    <div class="guide-card">
                        <h4>🐾 User Guide</h4>
                        <p>- Select the analysis you want to run.</p>
                        <p><b>Cognitive dysfunction prediction:</b><br>
                        - Complete all questionnaires about your dog. Depending on the dogs condition, you may need observe for 6 months and then answer 6 more followup questions.</p>
                        <p><b>Disease regression analysis:</b><br>
                        - Choose your dog breed. <br>
                        - Select a disease and choose either one or more variables.</p>
                    </div>
                    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CSLB PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "cslb":

    if st.button("← Back to Home"):
        go_home()
        st.rerun()

    st.header("🧠 Cognitive Dysfunction Prediction")
    st.caption("Prediction model for indicating the current state of your dogs brain")
    st.markdown("---")

    # ── Input form
    if st.session_state.cslb_predicted is None:

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🐾 Basic Information")
            dog_age    = st.number_input("Dog age (years)",  min_value=0,   max_value=18,    value=5,    step=1)
            dog_weight = st.number_input("Dog weight (lbs)", min_value=1.0, max_value=300.0, value=30.0, step=0.5)

            st.subheader("🏃 Activity & Health")
            activity_options = {
                "1 – Inactive / sedentary": 1,
                "2 – Somewhat active":      2,
                "3 – Moderately active":    3,
                "4 – Very active":          4,
                "5 – Extremely active":     5,
            }
            pa_activity = activity_options[
                st.selectbox("Activity level over the past year", list(activity_options.keys()))
            ]
            eye_val = 0 if st.selectbox("Eye health condition?", ["No", "Yes"]) == "No" else 2
            ear_val = 0 if st.selectbox("Ear health condition?", ["No", "Yes"]) == "No" else 2

        with col_right:
            st.subheader("🧠 Behavioural Questions (CSLB)")
            opts = {"Never (0)": 0, "Rarely (1)": 1, "Sometimes (2)": 2, "Often (3)": 3, "Always (4)": 4}
            keys = list(opts.keys())

            cslb_pace       = opts[st.selectbox("Paces back and forth without purpose?",              keys, key="cslb_pace")]
            cslb_stare      = opts[st.selectbox("Stares blankly into space or at walls?",             keys, key="cslb_stare")]
            cslb_stuck      = opts[st.selectbox("Gets stuck behind furniture or in corners?",         keys, key="cslb_stuck")]
            cslb_recognize  = opts[st.selectbox("Fails to recognise familiar people?",                keys, key="cslb_recognize")]
            cslb_walk_walls = opts[st.selectbox("Walks into walls or doors?",                         keys, key="cslb_walls")]
            cslb_avoid      = opts[st.selectbox("Avoids being petted or has lost interest?",          keys, key="cslb_avoid")]
            cslb_find_food  = opts[st.selectbox("Has difficulty finding food dropped on the floor?",  keys, key="cslb_food")]

        st.markdown("---")

        if st.button("🔍 Predict Cognitive Health", type="primary"):
            model    = train_cslb_model()
            input_df = pd.DataFrame([{
                "dd_age_years":             dog_age,
                "dd_weight_lbs":            dog_weight,
                "pa_activity_level":        pa_activity,
                "hs_health_conditions_eye": eye_val,
                "hs_health_conditions_ear": ear_val,
                "cslb_pace":                cslb_pace,
                "cslb_stare":               cslb_stare,
                "cslb_stuck":               cslb_stuck,
                "cslb_recognize":           cslb_recognize,
                "cslb_walk_walls":          cslb_walk_walls,
                "cslb_avoid":               cslb_avoid,
                "cslb_find_food":           cslb_find_food,
            }])

            st.session_state.cslb_predicted = float(model.predict(input_df)[0])
            st.session_state.cslb_inputs    = {
                "cslb_pace":       cslb_pace,
                "cslb_stare":      cslb_stare,
                "cslb_stuck":      cslb_stuck,
                "cslb_recognize":  cslb_recognize,
                "cslb_walk_walls": cslb_walk_walls,
                "cslb_avoid":      cslb_avoid,
                "cslb_find_food":  cslb_find_food,
            }
            st.rerun()

    # ── Results
    else:
        predicted = st.session_state.cslb_predicted
        inp       = st.session_state.cslb_inputs

        st.subheader("📊 Prediction Result")
        st.metric("Predicted CSLB Score", f"{predicted:.2f}")

        # ── HEALTHY ──────────────────────────────────────────────────────────
        if predicted < 40:
            st.success(
                f"✅ **Score {predicted:.1f} — No signs of cognitive dysfunction!**  \n"
                "Your dog appears cognitively healthy."
            )
            if st.button("🔄 Run another assessment"):
                go_cslb()
                st.rerun()

        # ── BORDERLINE → 6-month follow-up ───────────────────────────────────
        elif 40 <= predicted <= 60:
            st.warning(
                f"⚠️ **Score {predicted:.1f} — Possible early signs of cognitive dysfunction.**  \n"
                "Please complete the 6-month follow-up questions below."
            )
            st.markdown("---")
            st.subheader("🗓️ 6-Month Follow-Up Questions")
            st.info(
                "Answer these questions 6 months after the initial assessment. "
                "The final score is calculated using Dr. Hannah's CSLB formula from this paper (https://doi.org/10.1016/j.tvjl.2010.05.014)."
            )

            with st.form("followup_form"):
                fu_opts = {"Never (0)": 0, "Rarely (1)": 1, "Sometimes (2)": 2, "Often (3)": 3, "Always (4)": 4}
                fu_keys = list(fu_opts.keys())

                fu_col1, fu_col2 = st.columns(2)
                with fu_col1:
                    fu_pace_sel      = st.selectbox("(6mo) Paces back and forth?",              fu_keys, key="fu_pace_sel")
                    fu_stare_sel     = st.selectbox("(6mo) Stares blankly?",                    fu_keys, key="fu_stare_sel")
                    fu_defecate_sel  = st.selectbox("(6mo) Defecates/urinates in wrong place?", fu_keys, key="fu_defecate_sel")
                with fu_col2:
                    fu_food_sel      = st.selectbox("(6mo) Decreased interest in food?",        fu_keys, key="fu_food_sel")
                    fu_recognize_sel = st.selectbox("(6mo) Fails to recognise familiar people?",fu_keys, key="fu_recognize_sel")
                    fu_active_sel    = st.selectbox("(6mo) Less active or playful?",            fu_keys, key="fu_active_sel")

                submitted = st.form_submit_button("📋 Calculate Final CSLB Score", type="primary")

            if submitted:
                st.session_state.fu_pace      = fu_opts[fu_pace_sel]
                st.session_state.fu_stare     = fu_opts[fu_stare_sel]
                st.session_state.fu_defecate  = fu_opts[fu_defecate_sel]
                st.session_state.fu_food      = fu_opts[fu_food_sel]
                st.session_state.fu_recognize = fu_opts[fu_recognize_sel]
                st.session_state.fu_active    = fu_opts[fu_active_sel]
                st.session_state.fu_submitted = True
                st.rerun()

            # Final result — shown after follow-up submission ─────────────────
            if st.session_state.fu_submitted:
                actual_cslb = (
                    inp["cslb_pace"] + inp["cslb_stare"] + inp["cslb_stuck"] +
                    inp["cslb_recognize"] + inp["cslb_walk_walls"] +
                    inp["cslb_avoid"] + inp["cslb_find_food"] +
                    st.session_state.fu_pace +
                    st.session_state.fu_stare +
                    st.session_state.fu_defecate +
                    2 * st.session_state.fu_food +
                    3 * st.session_state.fu_recognize +
                    st.session_state.fu_active
                )
                st.markdown("---")
                st.subheader("📋 Final CSLB Score (6-Month Update)")
                st.metric("Actual CSLB Score", actual_cslb)
                if actual_cslb < 50:
                    st.success(
                        f"✅ **Score {actual_cslb} — No signs of cognitive dysfunction!**  \n"
                        "Your dog's actual CSLB score is within the healthy range."
                    )
                else:
                    st.error(
                        f"🚨 **Score {actual_cslb} — Symptoms of cognitive dysfunction detected.**  \n"
                        "Please contact your veterinarian as soon as possible."
                    )
                if st.button("🔄 Start over"):
                    go_home()
                    st.rerun()

        # ── HIGH ─────────────────────────────────────────────────────────────
        else:
            st.error(
                f"🚨 **Score {predicted:.1f} — Symptoms of cognitive dysfunction!**  \n"
                "Please contact your veterinarian as soon as possible."
            )
            if st.button("🔄 Run another assessment"):
                go_cslb()
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DISEASE REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "regression":
    if st.button("← Back to Home"):
        go_home()
        st.rerun()
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.header("🔬 Disease regression Analysis")
        st.caption("Statistical analysis & suggestions on how to reduce the affects of a disease based on selected variables.")
        breed = st.selectbox(
            "🐕 Select your dog's breed",
            options=["— select a breed —"] + breed_list,
            index=0,
        )
        st.markdown(f"**Breed:** `{breed}`")
        breed_summary = get_most_common_disease_for_breed(hles_data, breed)
        st.warning(breed_summary)
        st.markdown("Select a disease and variable group(s), then click Run Regression Analysis.")
        st.markdown("---")
        reg_col1, reg_col2 = st.columns([1, 3])

        with reg_col1:
            disease_choice = st.selectbox(
                "Select a disease",
                options=list(DISEASE_COL_MAP.keys()),
                format_func=lambda x: x.capitalize(),
            )
            variable_groups = st.multiselect(
                "Select variable group(s)",
                options=["diet", "physical_activity", "behavior", "environment"],
                default=["diet"],
                format_func=lambda x: x.replace("_", " ").capitalize(),
            )
            run_btn = st.button("🔍Run Regression Analysis", type="primary")

        with reg_col2:
            if run_btn:
                if not variable_groups:
                    st.warning("Please select at least one variable group.")
                else:
                    disease_col = DISEASE_COL_MAP[disease_choice]
                    if disease_col not in hles_data.columns:
                        st.error(f"Column `{disease_col}` not found in dataset.")
                    else:
                        all_results = {}
                        for vg in variable_groups:
                            with st.spinner(f"Running regression for **{vg.replace('_', ' ')}**…"):
                                try:
                                    sugg, findings = run_regression(disease_choice, vg, hles_data)
                                    all_results[vg] = ("ok", sugg, findings)
                                except Exception as e:
                                    import traceback
                                    all_results[vg] = ("err", [], str(traceback.format_exc()))
                        st.session_state.reg_results = {
                            "disease": disease_choice,
                            "groups":  all_results,
                        }

            if st.session_state.reg_results:
                res = st.session_state.reg_results
                st.subheader(f"Results for: **{res['disease'].capitalize()}** diseases:")
                for vg, result_tuple in res["groups"].items():
                    st.markdown(f"#### 📌 {vg.replace('_', ' ').capitalize()}")
                    status = result_tuple[0]
                    if status == "err":
                        st.error(f"Regression failed for **{vg}**:\n```\n{result_tuple[2]}\n```")
                    else:
                        _, sugg, findings = result_tuple
                        if not findings:
                            st.info("No statistically significant associations found (p ≥ 0.05).")
                        else:
                            t_find, t_sugg = st.tabs(["📊 Statistical Findings", "💡 Suggestions"])
                            with t_find:
                                for f in findings:
                                    st.markdown(f"- {f}")
                            with t_sugg:
                                for s in sugg:
                                    st.markdown(f"- {s}")
                    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "---\n"
    "**Analytic Avengers** · Data Science II Seminar · MSc ICE · "
    "Data sourced from [dogagingproject.org](https://dogagingproject.org)"
)