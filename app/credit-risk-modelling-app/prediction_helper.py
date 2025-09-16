import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'artifacts/model_data.joblib'

model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']


def prepare_df(age, income, loan_amount, loan_tenure_months, dpd,
               credit_utilization_ratio, residence_type, loan_purpose, loan_type,
               num_open_accounts, delinquency_ratio, avg_dpd_per_delinquency):

    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_purpose_Auto': 1 if loan_purpose == 'Auto' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        'loan_type_Secured': 1 if loan_type == 'Secured' else 0,

        # Corrected column name from 'year_at_current_address' to 'years_at_current_address'
        'years_at_current_address': 1,

        # other dummy variables with exact expected column names
        'number_of_dependants': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])

    # Make sure all cols_to_scale exist in df, otherwise drop missing from cols_to_scale
    missing_cols = [col for col in cols_to_scale if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in input data for scaling: {missing_cols}")
        # Remove missing columns from cols_to_scale for scaler.transform()
        scale_cols = [col for col in cols_to_scale if col in df.columns]
    else:
        scale_cols = cols_to_scale

    df[scale_cols] = scaler.transform(df[scale_cols])
    df = df[features]

    return df



def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability

    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score < 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating


def predict(age, income, loan_amount, loan_tenure_months, dpd,
            credit_utilization_ratio, residence_type, loan_purpose, loan_type,
            num_open_accounts, delinquency_ratio, avg_dpd_per_delinquency):

    input_df = prepare_df(age, income, loan_amount, loan_tenure_months, dpd,
                         credit_utilization_ratio, residence_type, loan_purpose, loan_type,
                         num_open_accounts, delinquency_ratio, avg_dpd_per_delinquency)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating


if __name__ == '__main__':
    age = 25
    income = 45000
    loan_amount = 20000
    loan_tenure_months = 24
    dpd = 5
    credit_utilization_ratio = 40
    residence_type = 'Owned'
    loan_purpose = 'Education'
    loan_type = 'Unsecured'
    num_open_accounts = 3
    delinquency_ratio = 10
    avg_dpd_per_delinquency = 7

    print(predict(age, income, loan_amount, loan_tenure_months, dpd,
                  credit_utilization_ratio, residence_type, loan_purpose, loan_type,
                  num_open_accounts, delinquency_ratio, avg_dpd_per_delinquency))


