import streamlit as st
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    lgbm = joblib.load('./model.pkl')   # 本地测试路径，根据pkl文件所在位置进行更改
    # lgbm = joblib.load('./lgbm.pkl')  # 上传到github所需路径，路径无需更改

    class Subject:
        def __init__(self, age, gender, calcium, glucose, creatinine, bun, aniongap, bicarbonate):
            self.age = age
            self.gender = gender
            self.calcium = calcium
            self.glucose = glucose
            self.creatinine = creatinine
            self.bun = bun
            self.aniongap = aniongap
            self.bicarbonate = bicarbonate

        def make_predict(self):
            subject_data = {
                "age": [self.age],
                "gender": [self.gender],
                "calcium": [self.calcium],
                "glucose": [self.glucose],
                "creatinine": [self.creatinine],
                "bun": [self.bun],
                "aniongap": [self.aniongap],
                "bicarbonate": [self.bicarbonate]
            }

            # Create a DataFrame
            df_subject = pd.DataFrame(subject_data)

            # Make the prediction
            prediction = lgbm.predict_proba(df_subject)[:, 1]
            adjusted_prediction = np.round(prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>The model predicts the risk of 90-day death is {adjusted_prediction} %</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            explainer = shap.Explainer(lgbm)
            shap_values = explainer.shap_values(df_subject)
            # 力图
            shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_subject.iloc[0, :], matplotlib=True)
            # 瀑布图
            # ex = shap.Explanation(shap_values[1][0, :], explainer.expected_value[1], df_subject.iloc[0, :])
            # shap.waterfall_plot(ex)
            st.pyplot(plt.gcf())

    st.set_page_config(page_title='AMI 90-Day Mortality')
    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>Predicting AMI 90-Day Mortality</h1>
                    <p class='intro'></p>
                </div>
                """, unsafe_allow_html=True)
    age = st.number_input("Age (years)", value=50)
    gender = st.selectbox("Gender (Female = 0, Male = 1)", [0, 1], index=0)
    calcium = st.number_input("Calcium (mg/dl)", value=9.0)
    glucose = st.number_input("Glucose (mg/dl)", value=100.0)
    creatinine = st.number_input("Creatinine (mg/dl)", value=1.0)
    bun = st.number_input("BUN (mg/dl)", value=12.0)
    aniongap = st.number_input("Aniongap (mmol/L)", value=10.0)
    bicarbonate = st.number_input("Bicarbonate (mmol/L)", value=25.0)

    if st.button(label="Submit"):
        user = Subject(age, gender, calcium, glucose, creatinine, bun, aniongap, bicarbonate)
        user.make_predict()


main()
