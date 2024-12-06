import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import csv

# Provided reviews dataset
reviews = [
    ("Liberty Bell", "Solemn experience, I did not expect this visit to be an emotional experience but as I read about the history of our country, visualized the excavation of Abraham Lincoln’s homestead an read the narratives of enslaved people. I felt both inspired but sad at the same time knowing my own roots run deeply within the very fabric of this place but so too did the blood and sweat. Truth be told, this was the most American I have ever felt."),
    ("Liberty Bell", "I loved visiting the Liberty Bell. It is free to view, just have to stand in a line that moved pretty quickly and then you will find yourself in a small museum that shares some history about the bell and showcases some smaller artifacts."),
    ("Liberty Bell", "After walking through the short gallery, you will come upon a beautiful room that features the Liberty Bell framed with a tall glass wall and an unobstructed view of Liberty Hall."),
    ("Liberty Bell", "The staff was friendly and the process was seamless. You can walk all the way around the bell to take in every detail."),
    ("Liberty Bell", "One thing to note is that, though, the bell is free to view anytime, you will need to purchase an advance ticket to tour the inside of Liberty Hall across the street!"),
    ("Liberty Bell", "The visit to the Liberty Bell was awesome!! There was barely any waiting time!! It was very impressive to see it in real life. I would recommend this place to anyone who goes to Philadelphia for either week or weekends!!"),
    ("Liberty Bell", "When I came, it was not super busy during a Thursday midday."),
    ("Liberty Bell", "I got through security really quickly, and then to see the actual bell, there was not a long line."),
    ("Liberty Bell", "The Liberty Bell is definitely a must-see and is very nicely preserved."),
    ("Liberty Bell", "Not a big history buff, but they did a nice job on telling the story of the bell leading up to the viewing. Probably was more interested in this than other historical interests in town. No ticket, just wait in line."),
    ("Liberty Bell", "Would be better if it was outside. Its a literal tourist trap. Cant really get too close to look at it without interrupting someones photo. The idea of it is nice. The history is nice. Its just too crowded."),
    ("Liberty Bell", "Some history about the Bell. You can also see it from outside through the window. Free. You need to go through security so no big bags, no food, etc. Usually, there is a line outside so consider the weather."),
    ("Liberty Bell", "This requires tickets. You can purchase the day you are there or even while standing in front of building. We were there in off season though so availability may differ in summer months. It’s about 45 minutes and you see 2 rooms. Standing only and 4 steps. If you love History, you will enjoy."),
    ("Liberty Bell", "I always get mixed feelings when I view stuff like this. I appreciate the fact that they are preserving history, but I don't feel that it's being done respectfully. There's way too much emphasis on the slavery part of it all. From pictures to videos. Constantly pushing the narrative of that's who we are. Consistently showcasing people of color in a negative and destitute state, rather than showing our accomplishments and triumphs, regardless of the attacks that are still presented on us today."),
    ("Liberty Bell", "I am disappointed in the lack of representation of my people and the whitewashing of history. Downplaying the vicious role that was and is still being orchestrated today by those who created the battle for justice."),
    ("Liberty Bell", "It’s cool to see but underwhelming. Thankfully it’s free, so points there!! It’s cool the view of independence hall right in the back but you really don’t spend more than like 5 minutes here. I attached the picture of the bell, that’s basically it y’all. Also they had high security here but the people working there were knowledgeable on Philadelphia history."),
    ("Liberty Bell", "The Liberty Bell in Philadelphia, Pennsylvania, is a truly remarkable and historically significant site. Steeped in the rich tapestry of American history, it stands as a powerful symbol of freedom and independence. The experience of witnessing this iconic bell, with its famous crack, is both awe-inspiring and educational. The surrounding exhibits provide a comprehensive understanding of the bell's role in shaping the nation. A visit to the Liberty Bell is a must for anyone seeking a deeper connection to America's past."),
    ("Liberty Bell", "The entire Historic District should be explored on foot only. The building in which the Liberty Bell is showcased is nice with a lawn and the experience starts with a history and significance of this great bell and how it was a symbol for liberty and freedom not just for Americans, but for many global situations."),
    ("Liberty Bell", "The entire trip at Liberty Bell would take about 40 minutes to an hour, culminating with a few moments at the Bell itself."),
    ("Liberty Bell","Your trip to Philadelphia is not complete without this."),
    ("Liberty Bell", "Excellent FREE piece of American History to explore. They have lots of plaques inside the visitors center describing how the bell was made, when it rang, where its traveled to, how the crack was formed, and more. Across the street from Independence Hall, close to George Washington’s old home, and right next to the Visitors Center, there’s lots to do in this little area!"),
    ("Liberty Bell", "It's neat to see the liberty bell. There were long lines to get in, but they moved quickly. Just watch out for the birds in the rafters if you're standing under them. The beginning of the exhibit is very congested but thins out as you move through. You will need to go through security to get in. It's definitely worth checking out."),
    ("Liberty Bell", "This historical location for the US. The line today wasn’t bad about 15 mins. I’ve seen the line as long as 1 hour. You will need to go through metal detector to get in. The agents had no patience for any questions so be prepared. Once inside, they give you so much info about this liberty bell. It will take about 20-30 mins to read through. And on the way out just stop by and take a photo with an actual bell. Fun time."),
    ("Liberty Bell", "What a piece of history. There is a ton of information as you walk in. Plenty of people working to answer any questions and explain some of the history. He bell is very well taken care of. There are also some remains of Washington's home to view. I'm glad we are still able to view such an important piece of American history."),
    ("Liberty Bell", "short line, not particularly busy, weather was perfect."),
    ("Liberty Bell", "I love the preservation of history! It was equally viewable all the way around. Plenty to read and look at while standing in line to see the bell itself! Make the time, I was in and out in less than 30 minutes. There was not much of a line at all. Total time in and out was less than an hour."),
    ("Independence Hall", "This is a piece of history! You’ve gotta check it out! I made zero plans ahead of time, just showed up on a weekend and it was super easy. Free. There was a short line with security - similar to an airport. Once your in, you can read some placards with historical passages, ask the staff about its history, etc. I went during the fall and it was an absolutely gorgeous experience. Also super kid friendly."),
    ("Independence Hall", "We were 2 minutes late to our tour because we entered through the front as opposed to the back, which is where they have the security screening to get in. We watched from a fence as our tour group entered the building. Once we went in through the back, the rangers were very friendly and accommodating, and we were able to be put on another tour. The rangers and volunteers were all helpful."),
    ("Independence Hall", "Definitely worth seeing. Loved the park ranger tour guide. She did a wonderful job and it was very motivating to listen to her story of the birth of our nation. Very easy to buy tickets online and go through security. Make sure to be on time for your tour. It’s free, it’s only $1 processing fee for the ticket."),
    ("Independence Hall", "If you want to actually have full access to Indepence Hall, you need to get on line & get a timed ticket. They charge you $1.00 for each ticket. I guess for a processing fee. I think it is Independence hall.gov sight. It is well worth going to see where our Declaration of Independence was negotiated and signed. Just putting yourself back in that time period & imagining what it was like. Without a timed ticket, you can still see the Hall of Congress, which is interesting as well. If you enjoy our history, you will really enjoy this. FYI: you do have to go through metal detectors & bags , purses etc..searched."),
    ("Independence Hall", "The tour was good, though given the overabundance of students present, it ended up heavily tailored to them. The ranger leading the tour was knowledgeable and gave a good idea of the room where it happened (as well as the room where it didn't). You must book in advance - the tickets get eaten up pretty quickly, and the queue was very stressful with a timed ticket trying to get in on time. Security guard on the way in was extremely rude, and frankly, people should be guarded against her, not the other way around."),
    ("Independence Hall", "A very informational tour all in 20 minutes. The National Parks Officers were very knowledgeable.  I thoroughly enjoyed the tour and highly recommend ."),
    ("Independence Hall", "Truly awesome, what history! The Park Rangers are awesome, Larry gave us a great tour at Congress Hall. I would consider this a must see attraction if you're visiting Philadelphia."),
    ("Independence Hall", "Great aspect of history. We enjoyed learning about our history. Very clean and the park Rangers were helpful."),
    ("Independence Hall", "Great place to visit if you're a history buff and want a bit of time to kill.  They give tours guided by National Park Rangers that you can book in advance ( I booked a 12 o'clock noon tour at 11:45) for $1. Very informative and full of insightful knowledge about our country's history."),
    ("Elfreth's Alley", "Cute alley with nice hidden small garden. The museum is small and interesting, however the clothing on display isn’t super authentic historically speaking. But it’s quaint and a nice stop in the historic area."),
    ("Elfreth's Alley", "Whenever I walk down Elfreth’s Alley I feel like I’ve been transported back to the 1700s. Every time I visit I walk through this short street with so much history. All of the houses feel and look historic. Literally making you feel you’ve been transported back in time. I just love Old City Philadelphia. Best place to visit!"),
    ("Elfreth's Alley", "You can feel old charms by walking around the alley. Just by entering the alley, you will say 'wow' how beautifully all the old houses are maintained. It is a great place to walk around and feel the history."),
    ("Elfreth's Alley", "Beautiful piece of history hidden within a modern city. Untouched by time."),
    ("Elfreth's Alley", "This is truly a gem in philadelphia and a historic spot and one of the oldest streets in the US. Take the time to walk through this spot and you will be transported back in time. Beautiful street and well maintained. It is free to walk the street but there’s a small museum in there that you can visit for a nominal fee. Please be mindful as people do live in those houses so ensure you respect their privacy."),
    ("Elfreth's Alley", "Charming Alley Historic Museum offers a picturesque journey to the 1700s with its well-preserved buildings. The quaint atmosphere is perfect for capturing timeless photos. Some locals still inhabit these historic structures. Free to explore, it’s a delightful and quick historical escapade."),
    ("Elfreth's Alley", "Little ans pretty street, well preserved. People still live here so be mindful.")
    
]

# Save reviews to CSV
csv_filename = "Sentiment_landmarks.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["landmark", "comment"])
    writer.writerows(reviews)

# Load reviews into a DataFrame
df = pd.read_csv(csv_filename)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Analyze sentiment for each review
df["sentiment_category"] = df["sentiment_score"].apply(
    lambda score: "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
)


# Calculate sentiment summary for each landmark
landmark_summary = df.groupby("landmark")["sentiment_category"].value_counts().unstack(fill_value=0)

# Display recommendations
print("Welcome to History Helper!")
print("\nHere are the available landmarks and their sentiment summaries:\n")
for landmark, summary in landmark_summary.iterrows():
    print(f"{landmark}:")
    for sentiment, count in summary.items():
        print(f"  {sentiment}: {count}")
    print()

# Allow user to choose a landmark
selected_landmark = input("Enter the name of the landmark you want to explore: ")

if selected_landmark in df["landmark"].unique():
    print(f"\nReviews and Sentiment for {selected_landmark}:\n")
    filtered_reviews = df[df["landmark"] == selected_landmark]
    for _, row in filtered_reviews.iterrows():
        print(f"- {row['comment']} (Sentiment: {row['sentiment_category']})")
else:
    print("Invalid selection. Please try again!")
