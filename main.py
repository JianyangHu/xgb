import numpy as np
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import xgboost as xgb
import pickle
import shap
import pandas as pd
app_ui = ui.page_fluid(
    ui.panel_title("XGBoost prediction & interpretability"),
    ui.layout_sidebar(
        ui.panel_sidebar(

            #####连续变量
            ui.h5({"class": "card-title mt-0"}, "Continuous variable"),
            ui.input_slider("Shot", "Shot", value=1, min=0, max=40, step=1),
            ui.input_slider("Offside", "Offside", value=1, min=0, max=15, step=1),
            ui.input_slider("AirDuelWon", "AirDuelWon", value=1, min=0, max=100, step=1),
            ui.input_slider("Corner", "Corner", value=1, min=0, max=20, step=1),
            ui.input_slider("CrossAcc", "CrossAcc", value=1, min=0, max=100, step=1),
            ui.input_slider("Foul", "Foul", value=1, min=0, max=40, step=1),
            ui.input_slider("Tackle", "Tackle", value=1, min=0, max=100, step=1),
            ui.input_slider("SprintD", "SprintD", value=1, min=0, max=4000, step=1),
            ui.input_slider("GroundDuelWon", "GroundDuelWon", value=1, min=0, max=20, step=1),
            ui.input_slider("Cross", "Cross", value=1, min=0, max=100, step=1),
            ui.input_slider("LSRD", "LSRD", value=1, min=70000, max=99999, step=1),
            ui.input_slider("TackleWon", "TackleWon", value=1, min=0, max=100, step=1),
            ui.input_slider("HSRE", "HSRE", value=1, min=100, max=700, step=1),
            ui.input_slider("MSRD", "MSRD", value=1, min=8000, max=25000, step=1),
            ui.input_slider("FwdPass", "FwdPass", value=1, min=0, max=600, step=1),
            # 情景变量
            ui.h5({"class": "card-title mt-0"}, "Discrete variable"),
            ui.input_radio_buttons("Location", "Location", {"0": "Away", "1": "Home"}),
            ui.input_radio_buttons("OppRank", "OppRank", {"1": "Weak", "0": "Strong"}),
            # ui.input_action_button("start", "Start"),

            
        ),
        ui.panel_main(
            ui.div(
                {"class": "card mb-3"},
                ui.div(
                    {"class": "card-body"},
                    ui.h5({"vd": "card-title mt-0"}, "Variable Declaration"),
                ),

                ui.div(
                    {"class": "card-footer"},
                    ui.h5({"class": "card-title mt-0"}, "每个特征介绍"),
                ),
            ),
            ui.div(
                {"class": "card mb-3"},
                ui.div(
                    {"class": "card-body"},
                    ui.h5({"class": "card-title mt-0"}, "Outcome"),
                ),
                ui.div(
                    {"class": "card-footer"},
                    ui.h5({"class": "card-title mt-0"}, "预测结果"),
                    ui.output_text_verbatim("result", placeholder=True),

                ),
            ),
            ui.div(
                {"class": "card mb-3"},
                ui.div(
                    {"class": "card-body"},
                    ui.h5({"class": "card-title mt-0"}, "Local interpretability"),
                ),
                ui.div(
                    {"class": "card-footer"},
                    ui.h5({"class": "card-title mt-0"}, "案例SHAP+lime分析"),
                    ui.output_plot("p1"),
                ),
            ),
        ),
    ),
)
#f=["Location","Shot","Offside","AirDuelWon","GroundDuelWon","Corner","CrossAcc","Foul","Tackle","SprintD","Cross","OppRank","LSRD","TackleWon","HSRE","MSRD","FwdPass"]


def server(input,output,session):
    @output
    @render.text
    # @reactive.event(input.start)
    async def result():
        model_xgb = pickle.load(open("best model.dat", "rb"))

        x=[input.Location(),input.Shot(),input.Offside(),input.AirDuelWon(), input.GroundDuelWon(),input.Corner(),
            input.CrossAcc(),input.Foul(),input.Tackle(),input.SprintD(),
            input.Cross(),input.OppRank(),input.LSRD(),
            input.TackleWon(),input.HSRE(),input.MSRD(),input.FwdPass(),
            ]
        x=np.array(x).astype(float)
    
        return f"预测获胜的概率为概率为{model_xgb.predict_proba(np.array(x).reshape(1,-1))[0][1]}"

    # @output
    # @render.plot
    # @reactive.event(input.start)
    # def p1():
    #     model_xgb = pickle.load(open("best model.dat", "rb"))
    #     f=["Location","Shot","Offside","AirDuelWon","GroundDuelWon","Corner","CrossAcc","Foul","Tackle","SprintD","Cross","OppRank","LSRD","TackleWon","HSRE","MSRD","FwdPass"]
    #     x=[input.Shot(),input.Offside(),input.AirDuelWon(), input.GroundDuelWon(),input.Corner(),
    #         input.CrossAcc(),input.Foul(),input.Tackle(),input.SprintD(),
    #         input.Cross(),input.Location(),input.OppRank(),input.LSRD(),
    #         input.TackleWon(),input.HSRE(),input.MSRD(),input.FwdPass(),
    #         ]
    #     x=np.array(x).astype(float)
    #     x=pd.DataFrame(x.reshape(1,-1),columns=f)

    #     explainer=shap.TreeExplainer(model_xgb)

    #     shap_values=explainer.shap_values(x)

    #     fig=shap.force_plot(explainer.expected_value, shap_values[0,:], x.iloc[0,:],link='logit',matplotlib=True)


    #     return fig



app = App(app_ui, server,debug=True)
