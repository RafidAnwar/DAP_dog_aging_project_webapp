# 🐕 DAP by Analytic Avengers

A web app that predicts canine cognitive dysfunction and analyses disease risk factors using 40,000+ samples from the [Dog Aging Project](https://dogagingproject.org).

Dog owners often struggle to detect early signs of cognitive dysfunction in aging pets or understand how specific factors contribute to common diseases, lacking simple tools for proactive care.
This project addresses these challenges through two core achievements: (1) delivering personalized cognitive dysfunction predictions via interactive questionnaires that track health changes over time (including 6-month follow-ups), and (2) providing tailored regression-based recommendations of the user’s dog breed to mitigate nine key physical diseases by analyzing user-selected variables. Leveraging over 40,000 samples from Dog Aging Project.org, it transforms research data into an accessible app that empowers owners with actionable insights for better canine health outcomes.

---

## What It Does

**Cognitive Dysfunction Prediction** — answer a short questionnaire about your dog and get an instant cognitive health score. If the result is borderline, a 6-month follow-up assessment is unlocked to track changes over time.

**Disease Regression Analysis** — select your dog breed and find out which disease is most common for your dog. Then select any of 9 diseases (cancer, cardiac, kidney, etc.) and explore how diet, physical activity, behaviour, or environment statistically affects disease odds and get personalized suggestions. 

## Running the App

**With Microsoft Azure webapp:**
https://dap-analytic-avengers.azurewebsites.net

**With Docker:**
```bash
docker run -p 8501:8501 rafidanwar/dap:latest
```
Then open **http://localhost:8501**

## Usage Guide
- Select the analysis you want to run.
#Cognitive dysfunction prediction:
- Complete all questionnaires about your dog. Depending on the dogs condition, you may need observe for 6 months and then answer 6 more followup questions.
#Disease regression analysis:
- Choose your dog breed.
- Select a disease and choose either one or more variables.

---

## Tech Stack

Python · Streamlit · Docker · Azure App Service
