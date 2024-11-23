# setup streamlit 
# install python + install libraries + run streamlit server 

# create new kernal : bullseye --> install python3.11.10
FROM python:3.11.10-bullseye  


# create new folder (app) on the kernal + my project 
WORKDIR /app 


# copy requirments.txt from my os to kernal
COPY requirments.txt /app/requirments.txt


# install libraries
RUN pip install -r /app/requirments.txt


# copy project files to kernal 
COPY . /app/

# open port to streamlit  , any port number 
EXPOSE 8501  

# default run command : start from entrypooint : cache [previous steps] 
ENTRYPOINT [ "streamlit" , "run" ]

# run command 
CMD [ "/app/app.py" , "--server.port=8501" ,"--server.address=0.0.0.0"]

