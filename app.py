import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import shap
import pandas as pd
import streamlit.components.v1 as components
import time
import openai
import os
# openai.log = "debug"
openai.api_key = "sk-4PXP8GvDe7XyZm696bA8TCzpv5ZhKXTHFdVPQKe8RUKA5j26"
openai.api_base = "https://api.chatanywhere.com.cn/v1"

class BaseVisualizer:
    pass 


def save_html(out_file, plot, full_html=True):
    """ Save html plots to an output file.
    
    Parameters
    ----------
    out_file : str or file
        Location or file to be written to
    plot : BaseVisualizer
        Visualizer returned by shap.force_plot()
    full_html : boolean (default: True)
        If True, writes a complete HTML document starting 
        with an <html> tag. If False, only script and div
        tags are included.
    """



    # assert isinstance(plot, BaseVisualizer), "save_html requires a Visualizer returned by shap.force_plot()."
    internal_open = False
    if type(out_file) == str:
        out_file = open(out_file, "w", encoding="utf-8")
        internal_open = True
    
    if full_html:
        out_file.write("<html><head><meta http-equiv='content-type' content='text/html'; charset='utf-8'>")
    
    out_file.write("<script>\n")

    # dump the js cod
    with open("bundle.js", "r", encoding="utf-8") as f:
        bundle_data = f.read()
    out_file.write(bundle_data)
    out_file.write("</script>")
    
    if full_html:
        out_file.write("</head><body>\n")

    out_file.write(plot.html())
    
    if full_html:
        out_file.write("</body></html>\n")

    if internal_open:
        out_file.close()




st.set_page_config(
   page_title="XGBoost prediction & interpretability",
   page_icon="⚽️",
   layout="wide",
   initial_sidebar_state="expanded",
)


####标题
st.title('XGBoost prediction & interpretability')

###加载模型
model_xgb = pickle.load(open("best model.dat", "rb"))

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: black;font-size:30px'>Select the value of the variables </h2>", unsafe_allow_html=True)
    st.markdown("""---""")
    st.markdown("<h2 style='text-align: center; color: black;font-size:20px'>Continuous variables </h2>", unsafe_allow_html=True)
    st.markdown("""---""")
    x1 = st.slider('Shot', 0, 40, 0)
    x2 = st.slider('Offside', 0, 15, 0)
    x3 = st.slider('AirDuelWon', 0, 100, 0)
    x4 = st.slider('Corner', 0, 20, 0)
    x5 = st.slider('CrossAcc', 0, 100, 0)
    x6 = st.slider('Foul', 0, 40, 0)
    x7 = st.slider('Tackle', 0, 100, 0)
    x8 = st.slider('SprintD', 0, 4000, 0)
    x9 = st.slider('GroundDuelWon', 0, 100, 0)
    x10= st.slider('Cross', 0, 100, 0)
    x11 = st.slider('LSRD', 70000, 99999, 0)
    x12 = st.slider('TackleWon', 0, 100, 0)
    x13 = st.slider('HSRE', 100, 800, 0)
    x14 = st.slider('MSRD', 8000, 25000, 0)
    x15 = st.slider('FwdPass', 0, 600, 0)
    #离散
    st.markdown("""---""")
    st.markdown("<h2 style='text-align: center; color: black;font-size:20px'>Discrete variables </h2>", unsafe_allow_html=True)
    st.markdown("""---""")
    
    x16 = st.radio(
        "Location",
        ('Away', 'Home'))

    x17 = st.radio(
        "OppRank",
        ('Weak', 'Strong'))
    

x16_=0 if x16=="Away" else 1
x17_=0 if x17=="Strong" else 1


x=[x16_,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x17_,x11,x12,x13,x14,x15]
x=np.array(x).astype(float)

st.markdown("""---""")
with st.expander("Variable introduction"):
    st.write('''

1.Shot: The number of shots taken by the team can indicate their attacking ability. A higher number of shots may increase the chances of scoring and winning the game.

2.Offside: The number of offside calls against the team may indicate their lack of coordination and timing in their attacking plays. This may decrease their chances of scoring and winning the game.

3.AirDuelWon: Winning more aerial duels can indicate the team's strength in the air and their ability to defend against high balls. This may increase their chances of preventing the opponent from scoring and winning the game.

4.Corner: Winning more corners can indicate the team's ability to create scoring opportunities from set pieces. This may increase their chances of scoring and winning the game.

5.CrossAcc: The accuracy of crosses can indicate the team's ability to deliver quality balls into the box. This may increase their chances of scoring and winning the game.

6.Foul: The number of fouls committed by the team may indicate their lack of discipline and control on the field. This may decrease their chances of winning the game, especially if they receive red cards or concede free kicks in dangerous areas.

7.Tackle: The number of tackles made by the team can indicate their defensive strength and ability to win back possession. This may increase their chances of preventing the opponent from scoring and winning the game.

8.SprintD: The total distance covered by the team in sprints can indicate their fitness level and work rate. This may increase their chances of maintaining their performance throughout the game and winning the game.

9.GroundDuelWon: Winning more ground duels can indicate the team's ability to win back possession and control the midfield. This may increase their chances of creating scoring opportunities and winning the game.

10.Cross: The total number of crosses attempted by the team can indicate their attacking intent and willingness to create scoring opportunities from wide areas. This may increase their chances of scoring and winning the game.

11.LSRD: The total distance covered by the team in low-speed running can indicate their ability to maintain possession and control the tempo of the game. This may increase their chances of creating scoring opportunities and winning the game.

12.TackleWon: The percentage of tackles won by the team can indicate their defensive efficiency and ability to win back possession. This may increase their chances of preventing the opponent from scoring and winning the game.

13.HSRE: The total number of high-speed runs executed by the team can indicate their ability to create scoring opportunities and break through the opponent's defense. This may increase their chances of scoring and winning the game.

14.MSRD: The total distance covered by the team in medium-speed running can indicate their ability to transition between defense and attack and maintain their shape on the field. This may increase their chances of controlling the game and winning the game.

15.FwdPass: The number of forward passes made by the team can indicate their attacking intent and willingness to take risks. This may increase their chances of creating scoring opportunities and winning the game.
    ''')

if st.button('Start'):
    

    if x11!=0:

        shap.initjs()
        f=["Location","Shot","Offside","AirDuelWon","GroundDuelWon","Corner","CrossAcc","Foul","Tackle","SprintD","Cross","OppRank","LSRD","TackleWon","HSRE","MSRD","FwdPass"]
        x=pd.DataFrame(x.reshape(1,-1),columns=f)
        explainer=shap.TreeExplainer(model_xgb)
        shap_values=explainer.shap_values(x)

        xx=shap.force_plot(explainer.expected_value, shap_values[0,:], x.iloc[0,:],link='logit')
        save_html('shap.html', xx)

        with open("shap.html","r") as f:
            html=f.read()

        with st.spinner('Predicting and interpreting...'):
            time.sleep(2)

        
        st.markdown("""---""")
        st.header("Prediction result:")
        st.write(f"The probability of winning is: {round(float(model_xgb.predict_proba(np.array(x).reshape(1,-1))[0][1]),2)}")
        st.markdown("""---""")
        st.header("SHAP local interpretation:")
        components.html(html)
        
        st.success("Done!")
        st.markdown("""---""")


        st.header("Let GPT help you!!!(about 30 seconds)")
        
        with st.spinner('GPT is telling you why...'):
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                                    messages=[{"role": "user", 
                                                                
                                                                "content": "下面是一个队伍一场足球比赛的各种技术数据,这个队伍的位置是{},他的对手是{}队,shot是{},offside是{},AirDuelWon是{},Corner是{},CrossAcc是{},Foul是{},Tackle是{},SprintD是{},GroundDuelWon是{},Cross是{},LSRD是{},TackleWon是{},HSRE是{},MSRD是{},FwdPass是{},但是我用机器学习预测这只队伍的胜率只有{},你可以从从这只队伍在本场比赛的指标的角度来分析这些指标是怎么影响最终胜率的吗,要结合我所给的具体数值?用英语回答,谢谢你!".format(x16,x17,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,round(float(model_xgb.predict_proba(np.array(x).reshape(1,-1))[0][1]),2))}]
                                                    )
    
            st.write(completion.choices[0].message.content)

            st.success("Done!")
            st.markdown("""---""")


    else:
        st.warning('Please select a value for each variable and click Start again', icon="⚠️")

    


    




    



