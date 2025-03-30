import streamlit as st
import base64
import pandas
import joblib
import numpy as np

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    #st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("C:/Users/SESA742087/Desktop/TripAdvisor2.jfif")
def getvisit_mode(mode):
    if mode=="Business":
        VisitModeId=1
    elif mode=="Couples":
         VisitModeId=2
    elif mode=="Friends":
        VisitModeId=4
    elif mode=="Family":
        VisitModeId=3
    else:
        VisitModeId=5
    return VisitModeId
def check_vacation(month):
    if month in [4,5,11,12]:
        return True
    else:
        return False



   

st.markdown("<h1 style 'color:white;'>TRIP ADVISOR</h1>",unsafe_allow_html=True) # setting the app title

t=st.sidebar.radio("## Navigation",["Rating Predictor","VisitMode Predictor","Vacation Selector"])
prediction=0

     
if t=="Rating Predictor":
    ratingpredictor_csv=pandas.read_csv("C:/Users/SESA742087/Documents/STREAMLIT/env/RaitingPredictorInputs.csv")
    ratingpredictor_df=pandas.DataFrame(ratingpredictor_csv)
    User_Id=st.text_input("Enter Your User Id")
    PlaceofVisit=st.text_input("Enter the destination to be visited")
    VisitDate=st.date_input("Enter the planned date of visit")
    visitmonth=VisitDate.month
    visityear=VisitDate.year
 

    if User_Id and PlaceofVisit and VisitDate :
            if PlaceofVisit in ratingpredictor_df['Attraction'].values:
                AverageRatingforAttraction=ratingpredictor_df.loc[ratingpredictor_df['Attraction']==PlaceofVisit,'AverageRatingforAttraction'].values[0]
                season_Summer=ratingpredictor_df.loc[(ratingpredictor_df['VisitMonth']==int(visitmonth)),'season_Summer'].values[0]
                season_Monsoon=ratingpredictor_df.loc[(ratingpredictor_df['VisitMonth']==int(visitmonth)),'season_Monsoon'].values[0]
                season_Spring=ratingpredictor_df.loc[(ratingpredictor_df['VisitMonth']==int(visitmonth)),'season_Spring'].values[0]
                season_Winter=ratingpredictor_df.loc[(ratingpredictor_df['VisitMonth']==int(visitmonth)),'season_Winter'].values[0] 
                if visityear in ratingpredictor_df.loc[ratingpredictor_df['Attraction']==PlaceofVisit,'VisitYear'].values:
                    yearly_growth_rate= ratingpredictor_df.loc[(ratingpredictor_df['Attraction']==PlaceofVisit)&(ratingpredictor_df['VisitYear']==int(visityear)),'yearly_growth_rate'].values[0]
                    previousyearrating= ratingpredictor_df.loc[(ratingpredictor_df['Attraction']==PlaceofVisit)&(ratingpredictor_df['VisitYear']==int(visityear)),'previousyearrating'].values[0]
                else:
                    attraction_df=ratingpredictor_df[ratingpredictor_df['Attraction']==PlaceofVisit]
                    previous_year=attraction_df['VisitYear'].iloc[(attraction_df['VisitYear'] - visityear).abs().argsort()[:1]].values[0]
                    previousyearrating=attraction_df.loc[attraction_df['VisitYear']==int(previous_year),'Rating'].values[0]
                    yearly_growth_rate=attraction_df.loc[attraction_df['VisitYear']==int(previous_year),'yearly_growth_rate'].values[0]

            else:
                 st.write("This ia a new attraction.We need to Update our Database .. Kindly try with some other destination until we update")

            if int(User_Id) in ratingpredictor_df['UserId'].values:      
                AveragrRatingbyUser=ratingpredictor_df.loc[ratingpredictor_df['UserId']==User_Id,'AveragrRatingbyUser'].values[0]
            else:
                AveragrRatingbyUser=ratingpredictor_df.loc[ratingpredictor_df['Attraction']==PlaceofVisit,'AverageRatingforAttraction'].values[0]

            if PlaceofVisit in ratingpredictor_df['Attraction'].values:
                 
                if st.button("Predict Rating"):
                    st.write("Fetching the prediction") 
                    input_data = {
                    'yearly_growth_rate':[yearly_growth_rate],
                    'AverageRatingforAttraction': [AverageRatingforAttraction],
                    'AveragrRatingbyUser': [AveragrRatingbyUser],
                    'season_Summer': [season_Summer],
                    'season_Monsoon': [season_Monsoon],
                    'season_Winter': [season_Winter],
                    'season_Spring': [season_Spring],
                    'previousyearrating':[previousyearrating]
                    }
                    input_df = pandas.DataFrame(input_data)
                    st.write("Loading the Model")
                    #Loading the model
                    model=joblib.load('C:/Users/SESA742087/Documents/STREAMLIT/env/ratingpredictorwithlessfeatures.pkl')
                    # Pass the DataFrame to the model
                    prediction = model.predict(input_df)
                    clipped_prediction=np.clip(prediction,1,5)

                    # Display the prediction
                    st.write("Predicted Rating:",clipped_prediction)
elif t=="VisitMode Predictor": 
    visitmodepredictor_csv=pandas.read_csv("C:/Users/SESA742087/Documents/STREAMLIT/env/visitmode_csv")
    visitmodepredictor_df=pandas.DataFrame(visitmodepredictor_csv)
    User_Id=st.text_input("Enter Your User Id")
    PlaceofVisit=st.text_input("Enter the destination to be visited")
    VisitMonth=st.text_input("Enter the Visit Month(enter a number between 1 to 12)")
    cityname=st.text_input("Enter your Present City")

    
    if User_Id and PlaceofVisit and VisitMonth and cityname:
            if PlaceofVisit in visitmodepredictor_df['Attraction'].values:

                AttractionId=visitmodepredictor_df.loc[visitmodepredictor_df['Attraction']==PlaceofVisit,'AttractionId'].values[0]
                season_Summer=visitmodepredictor_df.loc[visitmodepredictor_df['VisitMonth']==int(VisitMonth),'season_Summer'].values[0]
                season_Monsoon=visitmodepredictor_df.loc[visitmodepredictor_df['VisitMonth']==int(VisitMonth),'season_Monsoon'].values[0]
                season_Spring=visitmodepredictor_df.loc[visitmodepredictor_df['VisitMonth']==int(VisitMonth),'season_Spring'].values[0]
                season_Winter=visitmodepredictor_df.loc[visitmodepredictor_df['VisitMonth']==int(VisitMonth),'season_Winter'].values[0]
                previousvisitmodeforattraction=visitmodepredictor_df.loc[visitmodepredictor_df['Attraction']==PlaceofVisit,'previousvisitmodeforattraction'].values[0]
                IsVacationTime=check_vacation(int(VisitMonth))


            else:
                 st.write("This ia a new attraction.We need to Update our Database .. Kindly try with some other destination until we update")
   
            if int(User_Id) in visitmodepredictor_df['UserId'].values:      
                UserId=int(User_Id)
                CityId=visitmodepredictor_df.loc[visitmodepredictor_df['UserId']==UserId,'CityId'].values[0] 
                PreviousVisitModebyUser=visitmodepredictor_df.loc[visitmodepredictor_df['UserId']==UserId,'PreviousVisitModebyUser'].values[0]
                visitmodebyuserfromsamecity=visitmodepredictor_df.loc[visitmodepredictor_df['CityId']==CityId,'visitmodebyuserfromsamecity'].values[0]
            else:
                 UserId=int(User_Id)
                 CityId=visitmodepredictor_df.loc[visitmodepredictor_df['CityName']==cityname,'CityId'].values[0] 
                 PreviousVisitModebyUser=previousvisitmodeforattraction
                 visitmodebyuserfromsamecity=visitmodepredictor_df.loc[visitmodepredictor_df['CityId']==CityId,'visitmodebyuserfromsamecity'].values[0]
            if PlaceofVisit in visitmodepredictor_df['Attraction'].values:   
                if st.button("Predict VisitMode"):
                    st.write("Fetching the prediction") 
                    input_data_visitmode = {
                    'AttractionId': [AttractionId],
                    'IsVacationTime':[IsVacationTime],
                    'PreviousVisitModebyUser': [PreviousVisitModebyUser],
                    'previousvisitmodeforattraction':[previousvisitmodeforattraction],
                    'season_Summer': [season_Summer],
                    'season_Monsoon': [season_Monsoon],
                    'season_Winter': [season_Winter],
                    'season_Spring': [season_Spring],
                    'visitmodebyuserfromsamecity':[visitmodebyuserfromsamecity]
                    }
                    visitmodeinput_df = pandas.DataFrame(input_data_visitmode)
                    st.write("Loading the Model")
                    #Loading the model
                    model=joblib.load('C:/Users/SESA742087/Documents/STREAMLIT/env/visitmodepredictor.pkl')
                    # Pass the DataFrame to the model
                    prediction = model.predict(visitmodeinput_df)


                    # Display the prediction
                    st.write("Predicted Rating:",prediction)
else:
     recomender_csv=pandas.read_csv("C:/Users/SESA742087/Documents/STREAMLIT/env/Recomendationsystem")
     recomender_df=pandas.DataFrame(recomender_csv)
     user_item_matrix = recomender_df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
# Compute Pearson's correlation between users
     pearson_corr = user_item_matrix.T.corr()
# Function to get recommendations
     def get_recommendations(user_id, pearson_corr, user_item_matrix, data, top_n=3):
        similar_users = pearson_corr[user_id].sort_values(ascending=False).index[1:top_n+1]
        recommendations = user_item_matrix.loc[similar_users].mean(axis=0).sort_values(ascending=False)
        recommended_attractions = recommendations.index[:top_n]

    # Print the names of the recommended attractions
        attraction_names = []
        for rec_id in recommended_attractions:
            attraction_name = data.loc[data['AttractionId'] == rec_id, 'Attraction'].values[0]
            if attraction_name not in attraction_names:
                attraction_names.append(attraction_name)
        return attraction_names
     User_Id=st.text_input("Enter Your User Id")
     if User_Id:
        if st.button("Get Suggestions"):
            if int(User_Id) in recomender_df['UserId'].values:     
                    UserId=int(User_Id)
                    recommendations = get_recommendations(UserId, pearson_corr, user_item_matrix, recomender_df)
                    for attraction in recommendations:
                        st.write(attraction ,"\n")
            else:
                attraction_visits = recomender_df.groupby('AttractionId').size()

                    # Find the top 3 most popular attractions
                top_3_attractions = attraction_visits.nlargest(3)

                # Get the names of the top 3 attractions
                top_3_attraction_names = recomender_df[recomender_df['AttractionId'].isin(top_3_attractions.index)]['Attraction'].unique()

                st.write("The names of the top 3 attractions are:")
                for name in top_3_attraction_names:          
                    st.write(name)
     
                                         
                                     
                                     

                    

                
            
           
        
