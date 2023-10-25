#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import math
import argparse

### -----
from pathlib import Path
import streamlit as st


def convert_data_to_cvs(data_):
   return data_.to_csv(index=False).encode('utf-8')


def selected_data(raw_data_, start_sample_ = 0, stop_sample_ = 1):
    '''
    return pd data frame of the first nb_samples_   samples from raw_data_
    '''
    if start_sample_ < 0 or stop_sample_ < 0:
        raise ValueError("ERROR : samples nb can not be < 0 !!!")
    #lever une exception
    
    if start_sample_ > raw_data_.index.size or stop_sample_ > raw_data_.index.size:
        raise ValueError("ERROR : samples nb can not larger than duration !!!")
    
    part = raw_data_.iloc[start_sample_:stop_sample_,:]
    return part


def substract_bkg_level(raw_data_,bkg_level_):
    
    noiseless_data = pd.DataFrame(
        {
            "temps": raw_data_["temps"],
            "nb de coups": raw_data_["nb de coups"] - bkg_level_
        }
    )    
    return noiseless_data



### physics model functions
def one_exponential_model(x_,ampl0_,lambda0_):
    '''
    model is : f(x) = ampl0 * exp-(lambda0*x)
    '''
    # param_[0] = ampl0
    # param_[1] = lambda0

    values =  ampl0_ * np.exp(-lambda0_ * x_)
    return values


def two_exponential_model(x_,ampl0_,lambda0_,ampl1_,lambda1_):
    '''
    model is : f(x) = ampl0 * exp-(lambda0*x) + ampl1 * exp-(lambda1*x)
    '''
    # param_[0] = ampl0
    # param_[1] = lambda0
    # param_[2] = ampl1
    # param_[3] = lambda1
    
    values =  ampl0_ * np.exp(-lambda0_ * x_) + ampl1_ * np.exp(-lambda1_ * x_)
    return values


def linear_model(x_,a_,b_):
    '''
    model is : f(x) = a*x+b
    '''    
    # param_[0] = a
    # param_[1] = b

    values =  a_ * x_ + b_
    return values
### end of physics model functions


### process the fit
def fit_model(model_name_from_list_,data_,init_params_):

    if model_name_from_list_ == "exponential":
        best_vals, covar = curve_fit(one_exponential_model, data_["temps"], data_["nb de coups"], p0=init_params_, absolute_sigma=False)#,bounds=(0, [2*max(data_["nb de coups"]), 1.]))
        
    elif model_name_from_list_ == "linear":
        best_vals, covar = curve_fit(linear_model, data_["temps"], data_["nb de coups"], p0=init_params_, absolute_sigma=True)
        
    elif model_name_from_list_ == "two_exponentials":
        best_vals, covar = curve_fit(two_exponential_model, data_["temps"], data_["nb de coups"], p0=init_params_, absolute_sigma=True,bounds=(0, [2*max(data_["nb de coups"]), 1.,2*max(data_["nb de coups"]),1]))

    else:
        print("ERROR : This model is not on the known models list")
        #lever une exception

    return best_vals



def get_fitted_model(data_,model_name_,params_):
    if model_name_ == "exponential":
        fitted_model = one_exponential_model(data_['temps'],fitted_params[0],fitted_params[1])
        fitted_data = pd.DataFrame(
            {
                "temps": data_["temps"],
                "nb de coups": fitted_model,
            }
        )

    if model_name_ == "two_exponentials":
        fitted_model = two_exponential_model(data_['temps'],fitted_params[0],fitted_params[1],fitted_params[2],fitted_params[3])
        fitted_data = pd.DataFrame(
            {
                "temps": data_["temps"],
                "nb de coups": fitted_model,
            }
        )
    if model_name_ == "linear":
        fitted_model = linear_model(data_['temps'],fitted_params[0],fitted_params[1])
        fitted_data = pd.DataFrame(
            {
                "temps": data_["temps"],
                "nb de coups": fitted_model,
            }
        )
    return fitted_data

        
def draw_event_vs_time(data_,log, model_ = pd.DataFrame({'A' : []}),title_=None,output_filename_=None):
    '''
    log True  : log scaling on the y-axis.
    log False : linear scaling on the y-axis.
    '''
    if log == False:
        fig = plt.plot(data_["temps"],data_["nb de coups"],'*b',label='data')
        if model_.empty == False:
            fig = plt.plot(model_["temps"],model_["nb de coups"],'-r',label='model')
        fig =  plt.ticklabel_format(axis='both', style='sci', scilimits=(4,1))
    else:
        plt.semilogy(data_["temps"],data_["nb de coups"],'*b',label='data')
        if model_.empty == False:
            plt.semilogy(model_["temps"],model_["nb de coups"],'-r',label='model')

    fig =  plt.title(title_)
    fig =  plt.xlabel("temps")
    fig =  plt.ylabel("nb de coups")

    fig =  plt.grid()
    fig =  plt.legend()
    if output_filename_==None:
        fig =  plt.show()
        toto = st.write(fig)
    else:
        plt.savefig(output_filename_, bbox_inches='tight')




### main ####

separator = " "
columns_label = ["temps", "nb de coups"]
usecols=["temps", "nb de coups"] # j'utilise ces colonnes
skiprows = 1


apptitle = 'Mon premier streamlit'
st.set_page_config(page_title=apptitle, page_icon=":chart:")

st.header("Ma jolie entete")

input_file_option = st.selectbox(
   "Select your input file",
   ("raw_data_long.csv", "raw_data.csv"),
   index=None,
   placeholder="Input raw data file...",
)



st.write('You selected:', input_file_option)



st.sidebar.subheader('Interactive data selection')
st.sidebar.subheader('fit function selection')

fit_type = st.sidebar.radio(
   "select your fit function",
   ["exponential","two_exponentials","linear"],#horizontal=True,
   captions = ["exponential","two_exponentials","linear"])


st.sidebar.write('fit function is : ',fit_type)


if fit_type == "exponential":
   init_params = np.array([1500,0.1])
elif fit_type == "linear":
   init_params = np.array([-1,1500])
elif fit_type == "two_exponentials":
   init_params = np.array([1500,0.1, 20, 0.001])
else:
   init_params = None

st.sidebar.write('Init params:', init_params)



if input_file_option != None:
   input_file_option="./data/"+input_file_option
   input_path = input_file_option#"raw_data_long.csv"
   
   
   ###@st.cache_data
   data = pd.read_csv(input_path,
                      sep = separator,
                      skiprows = skiprows,
                      #index_col='temps',
                      header=None,names=columns_label,
                      usecols = usecols,
                      # names = header,
                      engine = 'python')

   
   st.sidebar.markdown("## option config")
   

   max_event = max(data["nb de coups"])
   max_time = data["temps"].index.size
   
   bkg_level = st.sidebar.slider('level of bkg?', min_value=0, max_value=max_event, step=1)
   st.sidebar.write("Bkg level is ", bkg_level)
   

   time_values = st.sidebar.slider(
      'Select a range of time',
      0, max_time, (0, max_time))

   st.sidebar.write('Values:', time_values)
    

        
   clean_data = substract_bkg_level(data,bkg_level)
   clean_data = selected_data(clean_data,time_values[0],time_values[1])
   
   
   fitted_params = fit_model(fit_type,clean_data,init_params)
   
   fitted_data = get_fitted_model(clean_data,fit_type,fitted_params)
   
   st.write('fitted parameters : ',fitted_params)
   st.write('T1/2 : ',math.log(2)/fitted_params[1], ' s')
   
   
   csv_data = convert_data_to_cvs(clean_data)
   
   st.download_button(
      "Download modified data as CSV",
      csv_data,
      "file.csv",
      "text/csv",
      key='download-csv'
   )
   


   st.subheader('Cleaned data display with fit')
   fig2,ax2  = plt.subplots()
   ax2.grid()
   ax2.plot(clean_data["temps"],clean_data["nb de coups"],'*b',label='data')
   ax2.plot(fitted_data["temps"],fitted_data["nb de coups"],'-r',label='data')
   st.pyplot(fig2)
   
   with st.expander("See notes"):
      
      st.markdown("""
      The N=f(t) plot shows the radioactive decay for an isotope.
      
      * The x-axis shows time
      * The y-axis shows Nb of decay / 10 s
      
      See also:
      
      * [Radon decay](https://sciencedemonstrations.fas.harvard.edu/presentations/radons-progeny-decay)
      * [RAdioactive decay](https://www.bfs.de/EN/topics/ion/environment/radon/introduction/introduction_node.html)
      
      """)
      
      
       
       
st.write("Jolie analyse des données de decay du Radon")
