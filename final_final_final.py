from distutils.sysconfig import get_python_inc, get_python_lib
# from sysconfig import get_python_version
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
# get_python_version().system('pip install tqdm')
import tkinter as tk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import datetime
from datetime import datetime, timedelta
from tqdm import tqdm
import os
from reportlab.lib.pagesizes import letter
from fpdf import FPDF
from datetime import datetime

tqdm.pandas()


def metrics(file_path):
    # file_path = "C:\\Users\\neeharika\\Documents\\Kairos\\isha_complete_bot_report_feed_prod_.xlsx"
    insights = []
    df = pd.read_excel(file_path)






    # 1. BOT INTERACTIONS

    value_counts = df['conversationId'].value_counts()

    # Convert 'createdAt' column to datetime data type
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Create a new column for grouping by specific hours
    df['hour_group'] = df['createdAt'].dt.strftime('%Y-%m-%d %H:00:00')

    # Group the data by 'hour_group' column
    grouped_data = df.groupby('hour_group')['conversationId'].size().reset_index(name='Number of Conversations')

    # Convert 'createdAt' column to datetime data type
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Create a new column for grouping by specific hours
    df['hour_group'] = df['createdAt'].dt.strftime('%H:00:00')

    # Group the data by 'hour_group' column
    grouped_data = df.groupby('hour_group')['conversationId'].size().reset_index(name='Number of Conversations')

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the line plot
    ax.plot(grouped_data['hour_group'], grouped_data['Number of Conversations'], color='blue', label='Line Plot')

    # Plot the bar plot
    ax.bar(grouped_data['hour_group'], grouped_data['Number of Conversations'],color='green', label='Bar Plot')

    # Set labels and title
    ax.set_xlabel('Time at which conversations created')
    ax.set_ylabel('Number of Conversations started')
    ax.set_title('Number of Conversations Started in a Bot in a particular time')

    # Rotate the x-axis tick labels by 90 degrees
    plt.xticks(rotation=90)

    # Add a legend
    ax.legend()

    # Display the plot
    plt.tight_layout()

    plt.savefig("fig_1_test.png")
    plt.close()

    # Find the max value and corresponding "created at hour"
    max_conversations = grouped_data['Number of Conversations'].max()
    max_hour = grouped_data.loc[grouped_data['Number of Conversations'] == max_conversations, 'hour_group'].iloc[0]

    insights.append("Number of conversations started by chat bot are :  " + str(
        df["conversationId"].nunique()) + '\n' + f"Max conversation count ({max_conversations}) is at ({max_hour})")






    # TOTAL NUMBER OF USERS

    total_numb_users = len(df['userId'].unique())





    # 2. BOT AVERAGE HANDLE TIME

    # Sorting the dataframe in ascending order with respect to 'createdAtUnixTime'
    sorted_df = df.sort_values('createdAtUnixTime')

    # Grouping the dataframe with respect to 'conversationId' and finding the difference between the first and last
    # instances of every conversation using 'createdAtUnixTime' feature and saving all of that as a series (result)
    result = sorted_df.groupby('conversationId')['createdAtUnixTime'].agg(lambda x: x.iloc[-1] - x.iloc[0])

    # Converting the values in the series from milliseconds to seconds and then to minutes
    result = result / 1000
    result = result / 60

    # Filtering out all the conversations which took place for only zero seconds
    filtered_result = result[result != 0]

    # Filtering out all the conversations which took place for over 60 minutes
    filtered_results = filtered_result[result <= 60]

    # Finding out the average Bot Handle Time for all the conversations except the ones which lasted for zero seconds
    filtered_results.mean()

    # Plotting the histogram
    sns.set_style("darkgrid")
    sns.histplot(filtered_results, bins=60, kde=True, edgecolor='black')
    plt.axvline(filtered_results.mean(), color='blue',
                linestyle='dashed', linewidth=1, label='Mean')
    plt.xlabel('Bot Handle Time (in minutes)')
    plt.ylabel('Number of Conversations')
    plt.title('Number of Conversations vs Bot Handle Time')
    plt.legend()
    plt.savefig("fig_2_test.png")
    plt.close()

    insights.append(f"Average Bot Handle Time is {filtered_results.mean():4.2f} minutes or {filtered_results.mean() * 60:.2f} seconds")






    # 3. BOT RESPONSE RATE

    numrow = df["message"].shape[0]
    df['response_count'] = df['botResponseText'].str.count('~') + 1
    Totalbotresponse_count = df['response_count'].sum()
    count = len(df[df['botResponseText'] == "Namaskaram! I am Isha Volunteer, your automated virtual assistant. How can I help you today? ~ Here are the options to help you get started."])
    bot_response  = Totalbotresponse_count - count

    ratio = count / bot_response

    # Create a list of labels for the pie chart
    labels = ['User Responses','Bot Responses (Excluding Welcome Message)']

    # Create a list of values for the pie chart
    sizes = [numrow, bot_response]
    
    plt.figure(figsize = (8, 8))
    
    # Create a pie chart
    plt.pie(sizes, labels=None, autopct='%1.1f%%')

    # Add a title to the pie chart
    plt.title('Ratio of User Responses to Bot Responses (Excluding Welcome Messages)')

    # Add a legend
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.7))
    plt.tight_layout()
    # Display the pie chart
    plt.savefig("fig_3_test.png", bbox_inches='tight')
    plt.close()
    insights.append(f"Ratio of User Responses to Bot Responses (excluding Welcome messages):  {ratio:.2f}")






    # 4. ENGAGEMENT RATE

    # Get the number of unique users who interacted with the bot
    # filtered_df = df[~df['message'].str.startswith('/')]

    num_unique_users=len(set(df['userId']))

    # Group by userId and conversationId, and count the number of unique transactionIds
    user_transaction_count = df.groupby(['userId', 'conversationId'])['transactionId'].nunique()

    # Filter the userIds with two or less than two transactionIds for every conversationId
    userIds_with_two_transactions = user_transaction_count.groupby('userId').filter(lambda x: x.max() <= 2).index.get_level_values('userId').unique()

    # Get the number of userIds with two or less than two transactionIds for every conversationId
    num_interactions = len(set(df['userId']))-len(userIds_with_two_transactions)

    percent_users_interacted = (num_interactions / num_unique_users) * 100



    # Create a pie chart to display the percentage of users who interacted with the bot
    labels = ['Users Interacted', 'Users Did Not Interact']
    sizes = [percent_users_interacted, 100 - percent_users_interacted]
    colors = ['#f2a900', '#cccccc']
    explode = (0.1, 0)


    fig1, ax1 = plt.subplots(figsize=(6, 6)) 
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=60)
    ax1.axis('equal')
    plt.title('Engagement rate')
    # Add a legend and adjust its position
    legend = plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    for text in legend.get_texts():
        text.set_color('red')  # Set the legend text color to red

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.savefig("fig_4_test.png")
    plt.close()
    insights.append(f"Percentage of users who interacted with the bot: {round(percent_users_interacted, 1)}%" +
                    '\n' + f"Percentage of users who did not interact with the bot: {round(100 - percent_users_interacted, 1)}%")






    # 5. RETURNING USERS

    df['createdAt'] = pd.to_datetime(df['createdAt'])
    df['createdAt'] = df['createdAt'].dt.date

    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Sort the DataFrame by 'createdAt' in ascending order
    df = df.sort_values('createdAt')

    # Set the threshold for 24 hours
    threshold = timedelta(hours=24)

    # Initialize a dictionary to store the count of occurrences for each user
    user_counts = {}

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        current_user_id = row['userId']
        current_created_at = row['createdAt']

        # Check if the user is already in the dictionary
        if current_user_id in user_counts:
            previous_created_at = user_counts[current_user_id]['last_created_at']

            # Calculate the time difference between current and previous query
            time_diff = current_created_at - previous_created_at

            # Check if the time difference is greater than the threshold
            if time_diff > threshold:
                # Increment the count of occurrences for the user and update the last_created_at
                user_counts[current_user_id]['count'] += 1
                user_counts[current_user_id]['last_created_at'] = current_created_at
        else:
            # Add the user to the dictionary with count = 1 and last_created_at = current_created_at
            user_counts[current_user_id] = {'count': 1, 'last_created_at': current_created_at}

    count_df = pd.DataFrame({'User ID': user_counts.keys(), 'Occurrence Count': [
                            value['count'] for value in user_counts.values()]})
    
    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the histogram plot
    ax.hist(count_df['Occurrence Count'], bins=10, edgecolor='black', color='skyblue')

    # Add labels and title
    ax.set_xlabel('Number of Queries', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)

    ax.set_title('Number of Users vs Number of Queries', fontsize=14)

    # Set the font size and style for the ticks
    ax.tick_params(axis='both', labelsize=10)


    plt.savefig("fig_5_test.png")
    # Display the plot
    plt.close()
    insights.append("     ")






    # 6. RETENTION RATE

    # # Calculate the total number of unique users
    # total_users = len(df['userId'].unique())

    # # Initialize a counter for returning users
    # returning_users = 0

    # # Iterate over the user_counts dictionary
    # for user_id, data in user_counts.items():
    #     # Check if the occurrence count is greater than 1
    #     if data['count'] > 1:
    #         returning_users += 1

    # # Calculate the percentage of returning users
    # percentage_returning_users = (returning_users / total_users) * 100

    # # Calculate the percentage of non-returning users
    # percentage_non_returning_users = 100 - percentage_returning_users
    
    # Convert 'createdAt' column to datetime
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Sort the DataFrame by 'createdAt' in ascending order
    df = df.sort_values('createdAt')

    # Set the threshold for 1 week
    threshold = timedelta(weeks=1)

    # Initialize a dictionary to store the count of occurrences for each user
    user_countss = {}

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        current_user_id = row['userId']
        current_created_at = row['createdAt']
        
        # Check if the user is already in the dictionary
        if current_user_id in user_countss:
            previous_created_at = user_countss[current_user_id]['last_created_at']
            
            # Calculate the time difference between current and previous query
            time_diff = current_created_at - previous_created_at
            
            # Check if the time difference is greater than the threshold
            if time_diff > threshold:
                # Increment the count of occurrences for the user and update the last_created_at
                user_countss[current_user_id]['count'] += 1
                user_countss[current_user_id]['last_created_at'] = current_created_at
        else:
            # Add the user to the dictionary with count = 1 and last_created_at = current_created_at
            user_countss[current_user_id] = {'count': 1, 'last_created_at': current_created_at}

    # Calculate the total number of unique users
    total_users = len(df['userId'].unique())

    # Initialize a counter for returning users
    returning_users = 0

    # Iterate over the user_counts dictionary
    for user_id, data in user_countss.items():
        # Check if the occurrence count is greater than 1
        if data['count'] > 1:
            returning_users += 1

    # Calculate the percentage of returning users within a week
    percentage_returning_users = (returning_users / total_users) * 100

    # Print the percentage of returning users within a week
    # print("Percentage of users that return to using the chatbot within a week:", percentage_returning_users)

    # Create a list of labels for the pie chart
    labels = ['Returning Users', 'Non-Returning Users']

    # Create a list of percentages for the pie chart
    sizes = [percentage_returning_users, 100 - percentage_returning_users]

    # Create a pie chart
    plt.pie(sizes, labels=None, autopct='%1.1f%%')

    # Add a title to the pie chart
    plt.title('Percentage of Returning Users vs Non-Returning Users')
    # Add a legend
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.7))
    plt.tight_layout()
    plt.savefig("fig_6_test.png", bbox_inches='tight')
    # Display the pie chart
    plt.close()
    insights.append(f"Percentage of users that return to using the chatbot within a week: {percentage_returning_users:.2f}%")







    # 7. NUMBER OF CLICKS

    # Categorize the values in the "isButtonClick" column
    df["action"] = df["isButtonClick"].apply(lambda x: "Button Click" if x in ["A", "Y"] else "User Typed")

    # Count the number of occurrences of each category and print the result
    action_counts = df["action"].value_counts()

    # Plot a pie chart of the categories
    labels = ['Button Clicks', 'User Typed']
    plt.figure(figsize=(8, 6))
    plt.pie(action_counts.values, labels = None, autopct='%1.1f%%')
    plt.title("Number of button clicks vs user typed messages")

    # Add a legend
    # plt.legend(labels, loc='center left', bbox_to_anchor=(0.8, 0.7))
    plt.legend(labels, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    # Display the pie chart
    plt.savefig("fig_7_test.png")
    plt.close()
    insights.append("    ")






    # 8. TOTAL NUMBER OF MESSAGES

    # Convert 'createdAt' column to datetime data type
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Create a new column for grouping by specific hours
    df['hour_group'] = df['createdAt'].dt.strftime('%Y-%m-%d %H:00:00')

    # Group the data by 'hour_group' column
    response_count = df.groupby('hour_group')['message'].size().reset_index(name='Total number of messages')

    # Convert 'createdAt' column to datetime data type
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Create a new column for grouping by specific hours
    df['hour_group'] = df['createdAt'].dt.strftime('%Y-%m-%d')

    # Group the data by 'hour_group' column
    response_count = df.groupby('hour_group')['message'].size().reset_index(name='Total number of messages')

    plt.figure(figsize=(12, 8))
    plt.bar(response_count['hour_group'], response_count['Total number of messages'])
    plt.xlabel('createdAt')
    plt.ylabel('message_count')
    plt.title('Total number of messages')
    plt.xticks(rotation=90)  # Rotate the x-axis labels by 90 degrees
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_8_test.png")
    plt.close()
    insights.append("    ")







    # 9. TOTAL BOT RESPONSES

    # Convert 'createdAt' column to datetime data type
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Create a new column for grouping by specific days
    df['day'] = df['createdAt'].dt.strftime('%Y-%m-%d')

    # Split 'botResponseText' column by '~' character and calculate the count of resulting segments
    df['response_count'] = df['botResponseText'].str.count('~') + 1

    # Group the data by 'day' column and sum the 'response_count' column
    response_count = df.groupby('day')['response_count'].sum().reset_index(name='Number of bot responses')

    # Filter data for the last 32 days
    # response_count_last_32_days = response_count.tail(32)

    # Plot the graph
    plt.figure(figsize=(12, 8))
    plt.bar(response_count['day'], response_count['Number of bot responses'])
    plt.xlabel('Day')
    plt.ylabel('Number of bot responses')
    plt.title('Number of bot responses per day')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_9_test.png")
    plt.close()
    insights.append("  ")






    # 10. MOST COMMON INTENTS

    # Get value counts and percentages
    counts = df['intentName'].value_counts().sort_values(ascending=False)
    index_0 = counts.index[0]
    percentages = counts / len(df) * 100
    # Group values below a certain percentage and label them as "Other"
    threshold = 5
    other_count = percentages[percentages < threshold].sum()
    counts = counts[percentages >= threshold]
    counts['Other'] = other_count
    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts.values, autopct='%1.1f%%')
    ax.set_title(f"Most frequently asked user queries")
    # Add legend
    ax.legend(counts.index, loc="best")
    # Style plot using Seaborn
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.savefig("fig_10_test.png")
    plt.close()
    insights.append(f"Most frequently asked user query: '{index_0}'")






    # 11. MOST COMMON FLOWS

    filtered_df = df[~df['message'].isin(['/chat-button-click'])]
    # filtered_df =filtered_df[~filtered_df['intentName'].isin(['ask_web_qa'])]
    # Filter out botResponseText containing "sorry"
    # filtered_df = filtered_df[~filtered_df['botResponseText'].str.contains('sorry', case=False)]# removing fall back querries
    filtered_df = filtered_df[filtered_df['isButtonClick'] == 'Y']
    # print("mess",filtered_df["message"].value_counts())
    # print(filtered_df["botResponseText"].head(10))# i wrote this to check wether sorry included columns deleted or not.
    # Count the frequency of each message
    msg_counts = filtered_df["message"].value_counts()
    # print("msg_counts:",msg_counts)
    # # Filter out messages with only one count
    # msg_counts = msg_counts[msg_counts > 1]\
    # Create a bar plot
    fig, ax = plt.subplots(figsize=(12, 9))
    msg_counts.plot(kind="bar", ax=ax)
    # Set the title and axis labels
    ax.set_title("Message Frequency")
    ax.set_xlabel("Message")
    ax.set_ylabel("Frequency")
    # Adjust the figure size and rotate the x-axis labels
    plt.tight_layout()  # Automatically adjusts the subplot parameters to fit the figure area
    plt.xticks(rotation=90)  # Rotate x-axis labels by 45 degrees
    plt.savefig("fig_11_test.png", bbox_inches='tight')
    plt.close()

    # Print the top 5 messages and their frequencies
    # print("Most triggered self served options by users are:")
    # Sort the message counts in descending order
    sorted_msg_counts = msg_counts.sort_values(ascending=False)
    # Select the top 5 messages
    top_5_messages = sorted_msg_counts.head(5)
    # for message, frequency in top_5_messages.items():
    #     print(f"Message: {message}, Frequency: {frequency}")
    insights.append(f"Top triggered option is '{top_5_messages.index[0]}' which was triggered {top_5_messages[0]} times.")






    # 12. MOST COMMON OPERATING SYSTEMS

    # Remove numbers and values with dots from the "sourcePlatformOS" column
    new_dfs = pd.DataFrame()
    new_dfs["sourcePlatformOS"] = df["sourcePlatformOS"].str.replace(r'\d+|-||,|', '', regex=True).str.replace("bit","",regex=True).replace('null nullnull', 'Empty').replace('', 'Empty').replace('null', 'Empty').str.replace("null", "",regex=True).str.replace("\.", "", regex=True)

    # Plot a graph for the OS categories
    os_count = new_dfs["sourcePlatformOS"].value_counts()
    plt.figure(figsize=(12, 6))
    plt.bar(os_count.index, os_count.values)
    plt.title("Most Used Operating System Categories")
    plt.xlabel("Types of Operating Systems")
    plt.ylabel("Number of Users")
    plt.tight_layout()

    # Adjust x-axis labels
    # Rotate labels by 45 degrees and align to the right
    plt.xticks(rotation=90)

    plt.savefig("fig_12_test.png", bbox_inches='tight')
    plt.close()
    # Get the most used OS platform
    most_used_os = os_count.idxmax()
    insights.append(f"The most used operating system platform is '{most_used_os.strip()}'")
    
    
    
    
    
    
    # 13. WELCOME ONLY

    # Group by userId and conversationId, and count the number of unique transactionIds
    user_transaction_count = df.groupby(['userId', 'conversationId'])['transactionId'].nunique()

    # Filter the userIds with two or less than two transactionIds for every conversationId
    userIds_with_two_transactions = user_transaction_count.groupby('userId').filter(lambda x: x.max() <= 2).index.get_level_values('userId').unique()

    # Get the number of userIds with two or less than two transactionIds for every conversationId
    num_users_with_two_transactions = len(userIds_with_two_transactions)

    # Calculate the percentage of users
    percentage_users_with_two_transactions = (num_users_with_two_transactions / len(set(df['userId']))) * 100

    # Print the result
    # print("Number of users who have not responded to the bot after a welcome message:", num_users_with_two_transactions)
    # print(f"Percentage of users who have not responded to the bot after a welcome message: {percentage_users_with_two_transactions:.2f}%")
    # print(f"Number of users who have responded to the bot after a welcome message: {len(set(df['userId'])) - num_users_with_two_transactions}")

    labels = ["Users who didn't respond after a welcome message", 'Users who responded after a welcome message']
    sizes = [num_users_with_two_transactions, len(set(df['userId'])) - num_users_with_two_transactions]  
    # Percentage of users for each type

    plt.figure(figsize=(8, 8))

    # Create the pie chart
    plt.pie(sizes, labels=None, autopct='%1.1f%%', startangle=0)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.7))


    # Apply Seaborn style
    sns.set()

    # Display the chart
    # plt.show()

    plt.savefig("fig_13_test.png", bbox_inches='tight')
    plt.close()
    insights.append(f"Number of non-responding users are : {num_users_with_two_transactions}" + '\n' + 
                    f"Number of responding users are : {len(set(df['userId'])) - num_users_with_two_transactions}")





    # 14. MOST COMMONLY USED BROWSERS

    # count of each browser
    browser_counts = df['sourcePlatformName'].value_counts()

    # bar chart of the browser counts
    browser_counts.plot(kind='bar')
    plt.xlabel('Browser')
    plt.ylabel('Count')
    plt.title('Most Commonly Used Browsers')

    # Add legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
    plt.tight_layout()
    plt.savefig("fig_14_test.png")
    plt.close()
    most_used_browser = browser_counts.idxmax()
    insights.append(f"The most commonly used browser is '{most_used_browser}'")
    
    
    
    
    
    # 15. CONFUSED STATE

    # Percentage of user queries to which the chatbot responded with a 'Sorry' message.
    sorry_percentage = df['botResponseText'].value_counts()[
        r"Sorry, I cannot help you with this. Please check our <a href='https://isha.sadhguru.org/us/en/center/isha-institute-inner-sciences-usa' target='_blank'>website</a> or contact support."] / len(
        df['botResponseText'])

    slices = [sorry_percentage, 1 - sorry_percentage]
    labels = ["Confused State (Chatbot responded with a 'Sorry' message)",
              "Chatbot did not give a 'Sorry' message"]
    explode = [0, 0.1]
    colors = sns.color_palette('pastel')[0:5]
    plt.figure(figsize=(8, 8))  # Increase the figure size

    plt.title('Pie Chart showing the % of user queries for which our Chatbot goes into a CONFUSED STATE')
    plt.pie(slices, labels=None, colors=colors, explode=explode,
            shadow=True, wedgeprops={'edgecolor': 'black'},  autopct='%1.1f%%')
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.7))  # Adjust the legend position

    # Adjust the bbox_inches parameter
    plt.savefig("fig_15_test.png", bbox_inches='tight')

    plt.close()
    insights.append(f"Percentage of user queries for which our Chatbot goes into a 'Confused' state: {(sorry_percentage * 100):.2f} %")

    
    
    
    
    
    # 16. ACCURACY
    
    new_df = pd.DataFrame(df[["message", "botResponseText"]])
    newdf = new_df.dropna(subset=['message', 'botResponseText'])
    
    # sample_df = newdf.sample(n = 1000)
    sorry_msg = "Sorry, I cannot help you with this. Please check our <a href='https://isha.sadhguru.org/us/en/center/isha-institute-inner-sciences-usa' target='_blank'>website</a> or contact support."
    USER_WHITELISTED_MESSAGES = ["Main menu", "Registered programs"]

    # Adjust the threshold as needed
    similarity_threshold = 0.225 

    from sentence_transformers import SentenceTransformer, util

    # Load a pre-trained Sentence Transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Function to check if the bot's response is correct using semantic search
    def is_response_correct(message, bot_response):
        if sorry_msg == bot_response:
            return 1
        
        for item in USER_WHITELISTED_MESSAGES:
            if item == message:
                return 3
        
        # Encode the message and bot's response into sentence embeddings
        message_embedding = model.encode(message, convert_to_tensor=True)
        response_embedding = model.encode(bot_response, convert_to_tensor=True)

        # Compute the cosine similarity between the embeddings
        similarity = util.cos_sim(message_embedding, response_embedding)

        # Return True if the cosine similarity is above a certain threshold, indicating a correct response
        if similarity.item() > similarity_threshold:
            return 3
        else:
            return 2

        # Add a new column to store the correctness of the bot's response
    newdf.loc[:, 'isResponseCorrect'] = newdf.progress_apply(lambda row: is_response_correct(row['message'], row['botResponseText']), axis=1)

    # Calculate the percentage of correct responses
    correct_count = (newdf['isResponseCorrect'] == 3).sum()
    sorry_count = (newdf['isResponseCorrect'] == 1).sum()
    total_count = len(newdf)
    percentage_correct = (correct_count / total_count) * 100
    percentage_sorry = (sorry_count / total_count) * 100


    slices = [percentage_correct, percentage_sorry, 100 - percentage_correct - percentage_sorry]
    labels = ["Bot has been accurate", "Fallback", "Bot has been inaccurate"]
    explode = [0, 0.1, 0.1]
    colors = sns.color_palette('pastel')[0:5]
    plt.figure(figsize=(8, 8))
    plt.title('Accuracy of our Chatbot')
    plt.pie(slices, labels = None, autopct='%1.1f%%', colors = colors, explode = explode, shadow = True, wedgeprops={'edgecolor': 'black'})
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.7))
    # plt.figure(figsize = (10, 8))

    plt.savefig("fig_16_test.png", bbox_inches='tight')
    plt.close()
    insights.append(f"Percentage of 'accurate' responses: {percentage_correct:.2f}%")
    # print(f"Bot Accuracy : {percentage_correct} %")




    # CONVERTING TO PDF
    
    # Initialize PDF object
    pdf = FPDF()
    pdf.add_page()

    # Add title
    pdf.set_font('Arial', 'B', 25)
    pdf.set_text_color(255, 0, 0)  # Set text color to red
    pdf.cell(0, 10, "Feed Report", 0, 1, 'C')
    pdf.ln(10)

    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Set the font and font size for the first page
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 0, 0)  # Reset to default (black)

    # Draw the current date and time on the first page
    pdf.cell(0, 10, "Generated on: {}".format(current_datetime), ln=True)
    pdf.ln(10)

    pdf.set_font('Arial', '', 14)
    pdf.set_text_color(0, 0, 0)  # Reset to default (black)
    pdf.multi_cell(0, 10, f"Number of columns : {df.shape[1]}", 0, 1)
    pdf.multi_cell(0, 10, f"Number of rows : {df.shape[0]}", 0, 1)
    pdf.multi_cell(0, 10, f"Total number of users : {total_numb_users}", 0, 1)
    pdf.ln(10)

    def save_plots_to_pdf():
        # Create a new PDF file
        pdf_filename = "feed_report.pdf"
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        plot_filenames = ["fig_1_test.png", "fig_2_test.png", "fig_3_test.png", "fig_4_test.png", "fig_5_test.png", "fig_6_test.png", "fig_7_test.png",
                          "fig_8_test.png", "fig_9_test.png", "fig_10_test.png", "fig_11_test.png", "fig_12_test.png", "fig_13_test.png", "fig_14_test.png", "fig_15_test.png", "fig_16_test.png"]

        # Define the list of headings
        headings = ["Bot Interactions: Total number of conversations started", "Bot Average Handle Time (AHT): \nThe average duration between first and last responses", "Bot Response Rate: \nRatio of user responses to the bot responses (excluding the welcome message)", "Engagement Rate: The percentage of users who interact with bot", "Returning Users: \nNumber of users that use bot for multiple queries at different points of time", "Retention Rate: \nPercentage of users that return to using the chatbot within the given time frame (One week)", "Number of Clicks: Number of button clicks vs user typed messages", "Total Messages: Total number of messages in all the conversations",
                    "Total Bot Responses: Total number of bot responses", "Most Common Intents: Most frequently asked users queries", "Most Common Flows: \nMost frequently triggered self-served options by users", "Most Used Operating Systems: \nUsage of different operating systems by users to access the bot", "Welcome Only: \nNumber of users  who  didnt respond to the bot after welcome message", "Most Commonly Used Browsers: \nUsage of different web browsers by users to access the bot", "Confused State: Percentage of user responses where the bot response is 'Sorry, I cannot help you with this'", "Accuracy: "]
        for index, filename in enumerate(plot_filenames):
            pdf.add_page()  # Add new page for each plot

            # Set the heading color
            pdf.set_text_color(165, 42, 42)  # Brown color

            # Set the font and size for the heading
            pdf.set_font('Arial', 'B', 16)

            # Add the heading
            pdf.multi_cell(0, 10, headings[index], 0, 1, 'C')

            # Reset the text color
            pdf.set_text_color(0, 0, 0)  # Reset to default (black)
            pdf.set_font('Arial', '', 16)
            # Rest of your code...
            pdf.image(filename, x=10, y=50, w=190)
            pdf.set_xy(10, 230)
            pdf.multi_cell(190, 10, txt=insights[index])
            pdf.ln(15)

            os.remove(filename)

        pdf.output(pdf_filename)

    save_plots_to_pdf()


if __name__ == "__main__":
    metrics()

# usegfoshlsehls