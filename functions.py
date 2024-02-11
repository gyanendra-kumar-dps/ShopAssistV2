import openai
import ast
import re
import pandas as pd
import json
import time


def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''
    
    delimiter = "####"
    example_user_req = {'GPU intensity': 'high','Display quality': 'high','Portability': 'low','Multitasking': 'high','Processing speed': 'high','Budget': '150000'}
    
    system_message = f"""

    You are an intelligent laptop gadget expert and your goal is to find the best laptop for a user.
    You need to ask relevant questions and understand the user profile by analysing the user's responses.
    You final objective is to fill the values for the different keys ('GPU intensity','Display quality','Portability','Multitasking','Processing speed','Budget') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this {{'GPU intensity': 'values','Display quality': 'values','Portability': 'values','Multitasking': 'values','Processing speed': 'values','Budget': 'values'}}
    The values for all keys, except 'budget', should be 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user. 
    The value for 'budget' should be a numerical value extracted from the user's response. 
    The values currently in the dictionary are only representative values. 
    
    {delimiter}Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised.
    - The values for all keys, except 'Budget', should strictly be either 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
    - The value for 'budget' should be a numerical value extracted from the user's response.
    - 'Budget' value needs to be greater than or equal to 25000 INR. If the user says less than that, please mention that there are no laptops in that range.
    - Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    {delimiter} Thought 1: Ask a question to understand the user's profile and requirements. \n
    If their primary use for the laptop is unclear. Ask another question to comprehend their needs.
    You are trying to fill the values of all the keys ('GPU intensity','Display quality','Portability','Multitasking','Processing speed','Budget') in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys. 
    Answer "Yes" or "No" to indicate if you understand the requirements and have updated the values for the relevant keys. \n
    If yes, proceed to the next step. Otherwise, rephrase the question to capture their profile. \n{delimiter}

    {delimiter}Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step. 
    Remember the instructions around the values for the different keys. Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same. 
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.{delimiter}

    {delimiter}Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary. 
    If you are not confident about any of the values, ask clarifying questions. {delimiter}

    Follow the above chain of thoughts and only output the final updated python dictionary. \n


    {delimiter} Here is a sample conversation between the user and assistant:
    User: "Hi, I am an editor."
    Assistant: "Great! As an editor, you likely require a laptop that can handle demanding tasks. Hence, the laptop should have high multitasking capability. You would also need a high end display for better visuals and editing. May I know what kind of work do you primarily focus on? Are you more involved in video editing, photo editing, or both? Understanding the specific type of editing work will help me tailor my recommendations accordingly. Let me know if my understanding is correct until now."
    User: "I primarily work with After Effects."
    Assistant: "Thank you for providing that information. Working with After Effects involves working with graphics, animations, and rendering, which will require high GPU. Do you work with high-resolution media files, such as 4K videos or RAW photos? Understanding your file sizes will help determine the storage capacity and processing power needed."
    User: "Yes, sometimes I work with 4K videos as well."
    Assistant: "Thank you for the information. Processing 4K vidoes will require a good processor and high GPU. I think we have already determined earlier that you need a high GPU. To ensure I have a complete understanding of your needs, I have one more question: Are you frequently on the go and require a laptop that is lightweight and easy to carry, or do you primarily work from a stationary location?"
    User: "Yes, sometimes I travel but do not carry my laptop."
    Assistant:"Could you kindly let me know your budget for the laptop? This will help me find options that fit within your price range while meeting the specified requirements."
    User: "my max budget is 1.5lakh inr"
    Assistant: "{example_user_req}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements. Do not start with Assistant: "
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation



def get_chat_model_completions(messages):
    #messages += ["Please format the code with ordered list."]
    print("format")
    print(messages)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
        max_tokens = 300
    )
    return response.choices[0].message["content"]



def moderation_check(user_input):
    print("moderation_check ....")
    print(user_input)
    time.sleep(22)
    response = openai.Moderation.create(input=user_input)
    print(response["results"])
    moderation_output = response["results"][0].flagged
    if moderation_output == True:
        return "Flagged"
    else:
        return "Not Flagged"


    
def intent_confirmation_layer(response_assistant):
    delimiter = "####"
    prompt = f"""
    You are a senior evaluator who has an eye for detail.
    You are provided an input. You need to evaluate if the input has the following keys: 'GPU intensity','Display quality','Portability','Multitasking',' Processing speed','Budget'
    Next you need to evaluate if the keys have the the values filled correctly.
    The values for all keys, except 'budget', should be 'low', 'medium', or 'high' based on the importance as stated by user. The value for the key 'budget' needs to contain a number with currency.
    Output a string 'Yes' if the input contains the dictionary with the values correctly filled for all keys.
    Otherwise out the string 'No'.

    Here is the input: {response_assistant}
    Only output a one-word string - Yes/No.
    """


    confirmation = openai.Completion.create(
                                    model="gpt-3.5-turbo-instruct",
                                    prompt = prompt,
                                    temperature=0)


    return confirmation["choices"][0]["text"]


check_dictionary_functions = [
    {
        'name': 'check_dictionary_functions',
        'description': 'Get the dictionary information from the body of the input text',
        'parameters': {
            'type': 'object',
            'properties': {
                'GPU intensity': {
                    'type': 'string',
                    'description': 'Name of the person'
                },
                'Display quality': {
                    'type': 'string',
                    'description': 'Major subject.'
                },
                'Portability': {
                    'type': 'string',
                    'description': 'The university name.'
                },
                'Multitasking': {
                    'type': 'string',
                    'description': 'GPA of the student.'
                },
                'Processing speed': {
                    'type': 'string',
                    'description': 'School club for extracurricular activities. '
                },
                'Budget': {
                    'type': 'string',
                    'description': 'School club for extracurricular activities. '
                }
                
            }
        }
    }
    
]

def dictionary_present(response):
    delimiter = "####"
    user_req = {'GPU intensity': 'high','Display quality': 'high','Portability': 'medium','Multitasking': 'high','Processing speed': 'high','Budget': '200000 INR'}
    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            Your task is to just extract and return only the python dictionary from the input.
            The output should match the format as {user_req}.
            The output should contain the exact keys and values as present in the input.

            Here are some sample input output pairs for better understanding:
            {delimiter}
            input: - GPU intensity: low - Display quality: high - Portability: low - Multitasking: high - Processing speed: medium - Budget: 50,000 INR
            output: {{'GPU intensity': 'low', 'Display quality': 'high', 'Portability': 'low', 'Multitasking': 'high', 'Processing speed': 'medium', 'Budget': '50000'}}

            input: {{'GPU intensity':     'low', 'Display quality':     'high', 'Portability':    'low', 'Multitasking': 'high', 'Processing speed': 'medium', 'Budget': '90,000'}}
            output: {{'GPU intensity': 'low', 'Display quality': 'high', 'Portability': 'low', 'Multitasking': 'high', 'Processing speed': 'medium', 'Budget': '90000'}}

            input: Here is your user profile 'GPU intensity': 'high','Display quality': 'high','Portability': 'medium','Multitasking': 'low','Processing speed': 'high','Budget': '200000 INR'
            output: {{'GPU intensity': 'high','Display quality': 'high','Portability': 'medium','Multitasking': 'high','Processing speed': 'low','Budget': '200000'}}
            {delimiter}

            Here is the input {response}

            """
    response = openai.ChatCompletion.create(
       # model="gpt-3.5-turbo-instruct",
        model="gpt-3.5-turbo",
        #prompt=prompt,
        messages = [{'role': 'user', 'content': prompt}],
        max_tokens = 2000,
        functions = check_dictionary_functions,
        function_call = 'auto'
        # temperature=0.3,
        # top_p=0.4
    )
    json_response = json.loads(response.choices[0].message.function_call.arguments)
    print(json_response)
    print(response["choices"][0])
    print(response.choices[0].message.function_call.arguments)
    #return response["choices"][0]["text"]
    return json.dumps(json_response)



def extract_dictionary_from_string(string):
    regex_pattern = r"\{[^{}]+\}"

    dictionary_matches = re.findall(regex_pattern, string)

    # Extract the first dictionary match and convert it to lowercase
    if dictionary_matches:
        dictionary_string = dictionary_matches[0]
        dictionary_string = dictionary_string.lower()

        # Convert the dictionary string to a dictionary object using ast.literal_eval()
        dictionary = ast.literal_eval(dictionary_string)
        return dictionary


def compare_laptops_with_user(user_req_object):
    laptop_df= pd.read_csv('updated_laptop.csv')
    print("test debug statement")
    #print(json.loads(user_req_string))
    #user_requirements = extract_dictionary_from_string(user_req_string)
    

    dictionary_string = str(user_req_object).lower()
    print(dictionary_string)
    dictionary = ast.literal_eval(dictionary_string)
    #user_requirements = extract_dictionary_from_string(json.loads(user_req_string))
    user_requirements = dictionary
    print("debug statements")
    print(user_requirements)
    budget = int(user_requirements.get('budget', '0').replace(',', ' ').split()[0])
    print(budget)
    #This line retrieves the value associated with the key 'budget' from the user_requirements dictionary.
    #If the key is not found, the default value '0' is used.
    #The value is then processed to remove commas, split it into a list of strings, and take the first element of the list.
    #Finally, the resulting value is converted to an integer and assigned to the variable budget.


    filtered_laptops = laptop_df.copy()
    filtered_laptops['Price'] = filtered_laptops['Price'].str.replace(',','').astype(int)
    filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= budget].copy()
    #These lines create a copy of the laptop_df DataFrame and assign it to filtered_laptops.
    #They then modify the 'Price' column in filtered_laptops by removing commas and converting the values to integers.
    #Finally, they filter filtered_laptops to include only rows where the 'Price' is less than or equal to the budget.

    mappings = {
        'low': 0,
        'medium': 1,
        'high': 2
    }
    # Create 'Score' column in the DataFrame and initialize to 0
    filtered_laptops['Score'] = 0
    for index, row in filtered_laptops.iterrows():
        user_product_match_str = row['laptop_feature']
        laptop_values = extract_dictionary_from_string(user_product_match_str)
        score = 0
        print("inside for loop")
        for key, user_value in user_requirements.items():
            if key.lower() == 'budget':
                continue  # Skip budget comparison
            laptop_value = laptop_values.get(key, None)
            laptop_mapping = mappings.get(laptop_value.lower(), -1)
            user_mapping = mappings.get(user_value.lower(), -1)
            print("laptop mapping comp")
            if laptop_mapping >= user_mapping:
                ### If the laptop value is greater than or equal to the user value the score is incremented by 1
                score += 1

        filtered_laptops.loc[index, 'Score'] = score

    # Sort the laptops by score in descending order and return the top 5 products
    top_laptops = filtered_laptops.drop('laptop_feature', axis=1)
    top_laptops = top_laptops.sort_values('Score', ascending=False).head(3)
    print("response records json")
    print(top_laptops.to_json(orient='records'))
    return top_laptops.to_json(orient='records')

def compare_function_calling(user_req_object):
    user_req_object = dictionary_present(user_req_object)
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": user_req_object}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "compare_laptops_with_user",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "GPU intensity": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "Display quality": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "Portability": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "Multitasking": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "Processing speed": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "Budget": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        
                    },
                    "required": ["user_req_object"],
                },
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "compare_laptops_with_user": compare_laptops_with_user,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                #user_req_string=function_args.get("user_req_string"),
                user_req_object=function_args,
                
            )
            
    print("funct resp")   
    print(function_response)    
    return function_response



def recommendation_validation(laptop_recommendation):
    data = json.loads(laptop_recommendation)
    data1 = []
    for i in range(len(data)):
        if data[i]['Score'] > 2:
            data1.append(data[i])

    return data1




def initialize_conv_reco(products):
    system_message = f"""
    You are an intelligent laptop gadget expert and you are tasked with the objective to \
    solve the user queries about any product from the catalogue: {products}.\
    You should keep the user profile in mind while answering the questions.\
    Start with a brief summary of each laptop in the following format, in decreasing order of price of laptops:
    1. <Laptop Name > : <Major specifications of the laptop > ,  <Price in Rs > \n
    2. <Laptop Name > : <Major specifications of the laptop > ,  <Price in Rs > \n
    
    """
    #1. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>
    #2. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>
    conversation = [{"role": "system", "content": system_message }]
    return conversation