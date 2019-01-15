#*** BUILDING AN INTERACTIVE CHATBOT***

#Import required modules
import pandas as pd
import random
import re
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from matplotlib import pyplot
from sklearn.svm import SVC
import spacy
import numpy as np
import sqlite3

"""
Part I - ELIZA technique 
"""
dav_said = "DAV : {0}"
you_said = "YOU : {0}"

#Defining the output to be displayed from bot using or concatinating the user message
def output(out):
    dav_message = "I can hear you! You said: " + out
    return dav_message

# Define a function that takes message to the bot: send_message and print both user and bot replies
def send_message(out):
    print(you_said.format(out))
    reply = output(out)
    print(dav_said.format(reply))

# Send a message to the bot
send_message("who arer you?")

"""
Here bot replies only if user msg is an exact match. 
Which in many cases is quite impossible, hence say it is a limitation. 
"""

# Define variables
name = "David"
connect = "customercare"

# Define a dictionary with the predefined replies
replies = {
  "what's your name?": "my name is {0}".format(name),
  "can you connect me to customercare?": "you will now be connected to {0}".format(connect),
  "default": "default message"
}

# Define the function that defines only the matched replies, else default reply
def output(out):
    if out in replies:
        dav_message = replies[out]
    else:
        dav_message = replies["default"]
    return dav_message

#Though its an exact match reply type, trying to make it interesting by setting list of replies to each question
name = "David"
music = "country"
climate = "sunny"
movie = "Bollywood"

# Define a dictionary containing a list of replies for each message
replies = {"what's your name?": [
        "my name is {0}".format(name),
        "they call me {0}".format(name),
        "I go by {0}".format(name)],
    "Can you play some music?": [
            "Here is the music you recently listened to {0}".format(music),
            "Playing you famous {0} music".format(music)],
    "what's today's weather?": [
            "the climate is {0}".format(climate),
            "today it is mostly {0}".format(climate),
            "it's {0} today".format(climate)],
  "default": ["default message"]
}

# Picking a random reply from list using random.choice for the user question
def output(out):
    if out in replies:
        dav_message = random.choice(replies[out])
    else:
        dav_message = random.choice(replies["default"])
    return dav_message

"""
ELIZA I: sending questions

"""

replies = {'question': ["I don't know :(", 'you tell me!'],
 'statement': ['tell me more!',
  'why do you think that?',
  'how long have you felt this way?',
  'I find that extremely interesting',
  'can you back that up?',
  'oh wow!',
  ':)']}

#output() function that defines what is question and what is not, and reply accordingly
def output(out):
    if out.endswith("?"):
        return random.choice(replies["question"])
    return random.choice(replies["statement"])


# Ends with a question mark
send_message("How are you?")
send_message("How are you?")

# No '?' at the end
send_message("Plan a vacation")
send_message("I think you are awesome!")


"""
ELIZA II: Extracting key phrases
"""

set_rules = {'I want (.*)': ['What would it mean if you got {0}',
  'Why do you want {0}',
  "What's stopping you from getting {0}"],
 'do you remember (.*)': ['Did you think I would forget {0}',
  "Why haven't you been able to forget {0}",
  'What about {0}',
  'Yes .. and?'],
 'do you think (.*)': ['if {0}? Absolutely.', 'No chance'],
 'if (.*)': ["Do you really think it's likely that {0}",
  'Do you wish that {0}',
  'What do you think about {0}',
  'Really--if {0}']}
 
# Define match_setrule() that looks for the defined pattern and choose a random reply if there is a match
def match_setrule(set_rules, out):
    reply, phrase = "default", None
    for pattern, replies in set_rules.items():
        pick_match = re.search(pattern, out)
        if pick_match is not None:
            reply = random.choice(replies)
            if '{0}' in reply:
                phrase = pick_match.group(1)
    return reply, phrase

# check if function is working as said
print(match_setrule(set_rules, "I want to go fishing"))

"""
ELIZA III: Pronouns
"""

# Define replace_pronouns()
def replace_pronouns(out):

    out = out.lower()
    if 'me' in out:
        return re.sub('I', 'you', out)
    if 'me' in out:
        return re.sub('me', 'you', out)
    if 'my' in out:
        return re.sub('my', 'your', out)
    if 'your' in out:
        return re.sub('your', 'my', out)
    if 'you' in out:
        return re.sub('you', 'me', out)

    return out

print(replace_pronouns("My project is exciting"))
print(replace_pronouns("remind me about the party"))
print(replace_pronouns("Do I own a Ranch"))

"""
ELIZA IV: Final step to gather all the functions and make it meaningful
"""

# Define output() to make it more lively call the match_setrule and in reply:replace pronouns and send reply
def output(out):
    reply, phrase = match_setrule(set_rules, out)
    if '{0}' in reply:
        phrase = replace_pronouns(phrase)
        reply = reply.format(phrase)
    return reply

# Send the messages
send_message("can you be my friend")
send_message("do you think we can go out together")
send_message("I am vey happy")

"""

PART - II:
    NLU-Natural Language Understanding(Is the main part of NLP)
    
                            When we look at a sentence:
                                /      \
                               /        \
                              /          \
(Intention of user to find)Intent           Entity(Details of the product requested)
to say it is search object  Eg:              Eg:
                            school              day
                            restaurant          name
                            airport             distance
"""


matched_words = {'restaurant': ['chinese', 'indian', 'mexican', 'thai'], 'direction': ['north', 'south', 'east', 'west'],
                 'greet': ['hello', 'hi', 'hey'], 'thankyou': ['thank', 'thx'], 
            'goodbye': ['bye', 'farewell', 'see you next time']}

replies = {'default': 'default message',
 'restaurant': 'you should try Indian cuisine',
 'goodbye': 'goodbye for now',
 'greet': 'Hello you! :)',
 'thankyou': 'you are very welcome',
 'direction': 'you are driving north'}

# Define a dictionary of patterns RegX compile method is used to match the pattern and join the values
patterns = {}
for intent, keys in matched_words.items():
    patterns[intent] = re.compile('|'.join(keys))
    
# Look at the patterns they became pattern objects whereas in previous step values are in list
print(patterns)

"""
Intent classification with regex  and entity extraction can be done 

spaCy
"""

# Load the original dataset and display some of the rows
air_tweets = pd.read_csv("E:/Fall 2018/Interview/ChatBot/Twitter_airlines_data.csv")
air_tweets.head()

# Retrieve only two columns which are going to be used later
# and additionally rename them
tweets = air_tweets[["airline_sentiment", "text"]]
tweets.columns = ("sentiment", "text", )

tweets.groupby("sentiment")\
      .size()\
      .reset_index(name="count")


def display_length_plot(tweets_df):
    """
    Displays a plot of tweets' lengths for given DataFrame.
    :param tweets_df: DataFrame
    """
    lengths = tweets_df["text"].str.len()\
                               .reset_index(name="length")\
                               .groupby("length")\
                               .size()\
                               .reset_index(name="count")
    lengths.plot(x="length", y="count", kind="line")
    
display_length_plot(tweets)

# Load the spacy model: nlp
nlp = spacy.load('en')

# Calculate the length of sentences
n_sentences = len(tweets)

# Calculate the dimensionality of nlp
embedding_dim = nlp.vocab.vectors_length

# Initialize the array with zeros: X
X = np.zeros((n_sentences, embedding_dim))

# Iterate over the sentences
for idx, sentence in enumerate(tweets):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X[idx, :] = doc.vector

"""
Intent classification with sklearn
"""

# Create a support vector classifier
clf = SVC()

# Fit the classifier using the training data
clf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = clf.predict(X_test)


"""
Using spaCy's entity recogniser: built-in entity recognizer to extract names, dates, and organizations 
from search queries.
"""
# Define included entities
include_entities = ['DATE', 'ORG', 'PERSON']

# Define extract_entities()
def extract_entities(out):
    # Create a dict to hold the entities
    ents = dict.fromkeys(include_entities)
    # Create a spacy document
    doc = nlp(out)
    for ent in doc.ents:
        if ent.label_ in include_entities:
            # Save interesting entities
            ents[ent.label_] = ent.text
    return ents

print(extract_entities('friends called Mary who have worked at Google since 2010'))
print(extract_entities('people who graduated from MIT in 1999'))

"""
Assigning roles using spaCy's parser
"""

# Create the document
doc = nlp("let's see that jacket in red and some blue jeans")

# Iterate over parents in parse tree until an item entity is found
def find_parent_item(word):
    # Iterate over the word's ancestors
    for parent in word.ancestors:
        # Check for an "item" entity
        if entity_type(parent) == "item":
            return parent.text
    return None

# For all color entities, find their parent item
def assign_colors(doc):
    # Iterate over the document
    for word in doc:
        # Check for "color" entities
        if entity_type(word) == "color":
            # Find the parent
            item =  find_parent_item(word)
            print("item: {0} has color : {1}".format(item, word))

# Assign the colors
assign_colors(doc) 


"""
Rasa_nlu is the package we are using 
"""

# Create args dictionary
args = {"key":"value"}
args = {"pipeline":"spacy_sklearn"}

# Create a configuration and trainer
config = RasaNLUConfig(cmdline_args=args)
trainer = Trainer(config)

# Load the training data
training_data = load_data("./training_data.json")

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

# Try it out
print(interpreter.parse("I'm looking for a Mexican restaurant in the North of town"))

"""
Since it is very small data extraction and entity recognition MITIE works well for short sentences. 
But when we decide NER on web scraped data I would say NLTK and SPACY shows better performance since
they have more categories when compared to MITIE
"""


pipeline = [
    "nlp_spacy",
    "tokenizer_spacy",
    "ner_crf"
]

# Create a config that uses this pipeline
args = {"key":"value"}
args = {"pipeline":pipeline}
config = RasaNLUConfig(cmdline_args = args)

# Create a trainer that uses this config
trainer = Trainer(config)

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

# Parse some messages
print(interpreter.parse("show me Chinese food in the centre of town"))
print(interpreter.parse("I want an Indian restaurant in the west"))
print(interpreter.parse("are there any good pizza places in the center?"))

"""
SQL statements in Python
to run a query against the hotels database to find all the expensive hotels in the south. 
"""

# Open connection to DB
conn = sqlite3.connect('hotels.db')

# Create a cursor
c = conn.cursor()

# Define area and price
area, price = "south", "hi"
t = (area, price)

# Execute the query
c.execute('SELECT * FROM hotels WHERE area=? AND price=?', t)

# Print the results
print(c.fetchall())

"""
Creating queries from parameters
Now implement a more powerful function for querying the hotels database.
"""
# Define find_hotels()
def find_hotels(params):
    # Create the base query
    query = 'SELECT * FROM hotels'
    # Add filter clauses for each of the parameters
    if len(params) > 0:
        filters = ["{}=?".format(k) for k in params]
        query += " WHERE " + " and ".join(filters)
    # Create the tuple of values
    t = tuple(params.values())
    
    # Open connection to DB
    conn = sqlite3.connect("hotels.db")
    # Create a cursor
    c = conn.cursor()
    # Execute the query
    c.execute(query, t)
    # Return the results
    c.fetchall()

# Create the dictionary of column names and values
params = {"area": "south", "price": "lo"}

# Find the hotels that match the parameters
print(find_hotels(params))

"""
Creating SQL from natural language
Now write a output() function which can handle messages like "I want an expensive hotel in the south of town" 
and respond appropriately according to the number of matching results in a database. This is important functionality 
for any database-backed chatbot.
"""

# Define output()
def output(out):
    # Extract the entities
    entities = interpreter.parse(out)["entities"]
    # Initialize an empty params dictionary
    params = {}
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find hotels that match the dictionary
    results = find_hotels(params)
    # Get the names of the hotels and index of the reply
    names = [r[0] for r in results]
    n = min(len(results),3)
    # Select the nth element of the replies array
    return replies[n].format(*names)

"""
Refining your search
Write a bot that allows users to add filters incrementally, in case they don't specify all of their 
preferences in one message.
"""

# Define a output function, taking the message and existing params as input

def output(out, params):
    # Extract the entities
    entities = interpreter.parse(out)['entities']
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find the hotels
    results = find_hotels(params)
    names = [r[0] for r in results]
    n = min(len(results), 3)
    # Return the appropriate reply
    return replies[n].format(*names), params

# Initialize params dictionary
params = {}
# Pass the messages to the bot
for out in ["I want an expensive hotel", "in the north of town"]:
    print("USER: {}".format(out))
    reply, params = output(out, params)
    print("BOT: {}".format(reply))