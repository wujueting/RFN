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
        def __init__(self, Age,Audiogram_type,ALB,Degree_of_hearing_loss,MCV,Onset_to_treatment):
            self.age = Age
            self.Audiogram_type = Audiogram_type
            self.ALB = ALB
            self.Degree_of_hearing_loss = Degree_of_hearing_loss
            self.MCV = MCV
            self.Onset_to_treatment = Onset_to_treatment
       

        def make_predict(self):
            subject_data = {
                "Age": self.Age,
                "Audiogram_type": self.Audiogram_type,
                "ALB": self.ALB,
                "Degree_of_hearing_loss": self.Degree_of_hearing_loss,
                "MCV": self.MCV,
                "Onset_to_treatment": self.Onset_to_treatment,
            }

            # Create a DataFrame
            df_subject = pd.DataFrame(subject_data)

            # Make the prediction
            prediction = lgbm.predict_proba(df_subject)[:, 1]
            adjusted_prediction = np.round(prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>The model predicts the hearing recovery rate is {adjusted_prediction} %</b>
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

    st.set_page_config(page_title='Hearing recovery of sudden sensorineural hearing loss')
    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>Predicting hearing recovery rate</h1>
                    <p class='intro'></p>
                </div>
                """, unsafe_allow_html=True)
    Age = st.number_input("Age (years)", value=50.5)
    Audiogram_type = st.selectbox("Audiogram_type (Ascending = 0, flat = 1, descending= 2, profound= 3 )", [0, 1, 2 , 3 ], index=0)
    ALB = st.number_input("ALB (g/dl)", value=44.75)
    Degree_of_hearing_loss = st.selectbox("Degree_of_hearing_loss (Normal = 0, Mild = 1, Moderate= 2, Moderately severe= 3 ,Severe=4,  Profound=5, Complete=6 )", [0, 1, 2 , 3 ,4 , 5, 6 ], index=0)
    MCV = st.number_input("MCV (fl)", value=88.85)
    Onset_to_treatment = st.number_input("Onset_to_treatment (days)", value=15.0)
    
    if st.button(label="Submit"):
        user = Subject(Age,Audiogram_type,ALB,Degree_of_hearing_loss,MCV,Onset_to_treatment)
        user.make_predict()


main()
