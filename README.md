# WAU Target Marketing

### Presentation link

https://docs.google.com/presentation/d/1rD321qPs3voNtAK6ZmwXp4MqcawSqLI0lc75BvN5DYQ/edit#slide=id.gdb927efea2_0_8

### Problem Statement

The marketing team at WAU Bank would like to better target their audience to promote advertisements on various forums that are more appropriate to the target audience. Using Natural Language Processing on Reddit I will be creating a Classification Model to distinguish between the r/HomeImprovement and r/StudentLoan subreddits. This model could be usable on other bodies of text to determine whether the user may be in a state of homeownership or student debt and whether to market Home Equity Loans or Mortgages to those users. 

### Data Sources

Student Loan Subreddit: https://www.reddit.com/r/StudentLoans/
Home Improvement Subreddit: https://www.reddit.com/r/HomeImprovement/

### Data Dictionary

**Individual Columns**
|Feature|Type|Dataframe|Description|
|--------|-----------|-------|------|
|ids|object|cldf|Reddit user ID|
|title|object|cldf|Title of the Post|
|body|object|cldf|Body of the subreddit post|
|comments|int64|cldf|Number of comments on the post|
|upvotes|int64|cldf|Number of upvotes on the post|
|subreddit|int64|cldf|Subreddit, r/StudentLoans = 1, r/HomeImprovement = 0| 
|post_length|int64|cldf|Overall number of characters in post|
|sentiment|float64|cldf|Sentiment of post, closer to -1 being negative, closer to 1 being more positive, 
|word_count|int64|cldf|Number of words in the post|

**Parts of speech measured in frequency of use**
|Feature|Type|Dataframe|Description|
|--------|-----------|-------|------|
|pos_PRON|float64|cldf|Parts of speech Pronouns|
|pos_VERB|float64|cldf|Parts of speech Verb|
|pos_DET|float64|cldf|Parts of speech Determinants|
|pos_ADJ|float64|cldf|Parts of speech Adjectives|
|pos_NOUN|float64|cldf|Parts of speech Noun|
|pos_CCONJ|float64|cldf|Parts of speech Coordinating Conjugation|
|pos_PART|float64|cldf|Parts of speech Particle|
|pos_PUNCT|float64|cldf|Parts of speech Punctuation|
|pos_ADP|float64|cldf|Parts of speech Adposition|
|pos_AUX|float64|cldf|Parts of speech Auxiliary|
|pos_ADV|float64|cldf|Parts of speech Adverb|
|pos_SPACE|float64|cldf|Parts of speech Space between characters|
|pos_INTJ|float64|cldf|Parts of speech Interjection|
|pos_PROPN|float64|cldf|Parts of speech Proper Noun|
|pos_X|float64|cldf|Parts of speech Other, unclassifiable ex. 'sdfjfhiuhdn'|
|pos_NUM|float64|cldf|Parts of speech Number|
|pos_SYM|float64|cldf|Parts of speech Symbol|
|pos_SCONJ|float64|cldf|Parts of speech Subordinating Conjugation|

**Lemmatized body column and word counts in post**
|Feature|Type|Dataframe|Description|
|--------|-----------|-------|------|
|lemma_body|object|cldf|Lemmatized version of body column for word analysis|
|able|float64|cldf|Frequently used word "able"|
|apply|float64|cldf|Frequently used word "apply"|
|appreciate|float64|cldf|Frequently used word "appreciate"|
|area|float64|cldf|Frequently used word "area"|
|ask|float64|cldf|Frequently used word "ask"|
|build|float64|cldf|Frequently used word "build"|
|close|float64|cldf|Frequently used word "close"|
|college|float64|cldf|Frequently used word "college"|
|company|float64|cldf|Frequently used word "company"|
|cost|float64|cldf|Frequently used word "cost"|
|cover|float64|cldf|Frequently used word "cover"|
|credit|float64|cldf|Frequently used word "credit"|
|currently|float64|cldf|Frequently used word "currently"|
|day|float64|cldf|Frequently used word "day"|
|debt|float64|cldf|Frequently used word "debt"|
|degree|float64|cldf|Frequently used word "degree"|
|door|float64|cldf|Frequently used word "door"|
|end|float64|cldf|Frequently used word "and"|
|federal|float64|cldf|Frequently used word "federal"|
|feel|float64|cldf|Frequently used word "feel"|
|fix|float64|cldf|Frequently used word "fix"|
|floor|float64|cldf|Frequently used word "floor"|
|graduate|float64|cldf|Frequently used word "graduate"|
|hi|float64|cldf|Frequently used word "hi"|
|high|float64|cldf|Frequently used word "high"|
|income|float64|cldf|Frequently used word "income"|
|instal|float64|cldf|Frequently used word "instal"|
|issue|float64|cldf|Frequently used word "issue"|
|job|float64|cldf|Frequently used word "job"|
|leave|float64|cldf|Frequently used word "leave"|
|light|float64|cldf|Frequently used word "light"|
|line|float64|cldf|Frequently used word "line"|
|live|float64|cldf|Frequently used word "live"|
|ll|float64|cldf|Frequently used word "ll"|
|long|float64|cldf|Frequently used word "long"|
|lot|float64|cldf|Frequently used word "lot"|
|low|float64|cldf|Frequently used word "low"|
|money|float64|cldf|Frequently used word "money"|
|option|float64|cldf|Frequently used word "option"|
|paint|float64|cldf|Frequently used word "paint"|
|parent|float64|cldf|Frequently used word "parent"|
|place|float64|cldf|Frequently used word "place"|
|possible|float64|cldf|Frequently used word "possible"|
|post|float64|cldf|Frequently used word "post"|
|private|float64|cldf|Frequently used word "private"|
|program|float64|cldf|Frequently used word "program"|
|rate|float64|cldf|Frequently used word "rate"|
|remove|float64|cldf|Frequently used word "remove"|
|room|float64|cldf|Frequently used word "room"|
|run|float64|cldf|Frequently used word "run"|
|small|float64|cldf|Frequently used word "small"|
|tell|float64|cldf|Frequently used word "tell"|
|tile|float64|cldf|Frequently used word "tile"|
|total|float64|cldf|Frequently used word "total"|
|water|float64|cldf|Frequently used word "water"|
|window|float64|cldf|Frequently used word "window"|
|wonder|float64|cldf|Frequently used word "wonder"|
|wood|float64|cldf|Frequently used word "wood"|

**Frequently used terms by number of uses in post**
|Feature|Type|Dataframe|Description|
|--------|-----------|-------|------|
|the_wall|float64|cldf|Frequently used term "the wall"|
|a_lot|float64|cldf|Frequently used term "a lot"|
|my_parent|float64|cldf|Frequently used term "my parent"|
|any_advice|float64|cldf|Frequently used term "any advice"|
|the_floor|float64|cldf|Frequently used term "the floor"|
|a_house|float64|cldf|Frequently used term "a house"|
|the_door|float64|cldf|Frequently used term "the door"|
|my_question|float64|cldf|Frequently used term "my question"|
|any_help|float64|cldf|Frequently used term "any help"|
|the_basement|float64|cldf|Frequently used term "the payment"|
|any_idea|float64|cldf|Frequently used term "any idea"|
|the_water|float64|cldf|Frequently used term "the water"|
|the_end|float64|cldf|Frequently used term "the end"|
|the_time|float64|cldf|Frequently used term "the time"|
|the_money|float64|cldf|Frequently used term "the money"|
|any_suggestion|float64|cldf|Frequently used term "any suggestion"|
|grad_school|float64|cldf|Frequently used term "grad school"|
|the_rest|float64|cldf|Frequently used term "the rest"|
|no_idea|float64|cldf|Frequently used term "no idea"|
|the_window|float64|cldf|Frequently used term "the window"|
|the_problem|float64|cldf|Frequently used term "the problem"|
|the_process|float64|cldf|Frequently used term "the process"|
|the_bottom|float64|cldf|Frequently used term "the bottom"|
|my_wife|float64|cldf|Frequently used term "my wife"|
|the_school|float64|cldf|Frequently used term "the school"|
|the_tile|float64|cldf|Frequently used term "the title"|
|#_x200b|float64|cldf|Frequently used term "# x200b"|
|the_top|float64|cldf|Frequently used term "the top"|
|the_ceiling|float64|cldf|Frequently used term "the ceiling"|
|the_attic|float64|cldf|Frequently used term "the attic"|
|the_side|float64|cldf|Frequently used term "the side"|
|this_point|float64|cldf|Frequently used term "this point"|
|the_roof|float64|cldf|Frequently used term "the roof"|
|the_room|float64|cldf|Frequently used term "the room"|
|a_job|float64|cldf|Frequently used term "a job"|
|the_cost|float64|cldf|Frequently used term "the cost"|
|the_wood|float64|cldf|Frequently used term "the wood"|
|the_good_way|float64|cldf|Frequently used term "the good way"|
|the_kitchen|float64|cldf|Frequently used term "the kitchen"|
|Sallie_Mae|float64|cldf|Frequently used term "Sallie Mae"|
|the_garage|float64|cldf|Frequently used term "the garage"|
|the_hole|float64|cldf|Frequently used term "the hole"|
|the_drywall|float64|cldf|Frequently used term "the drywall"|
|the_bathroom|float64|cldf|Frequently used term "the bathroom"|
|the_issue|float64|cldf|Frequently used term "the issue"|
|my_school|float64|cldf|Frequently used term "my school"|
|my_husband|float64|cldf|Frequently used term "my husband"|
|the_vent|float64|cldf|Frequently used term "the vent"|
|the_year|float64|cldf|Frequently used term "the year"|
|a_bit|float64|cldf|Frequently used term "a bit"|
|the_previous_owner|float64|cldf|Frequently used term "the previous owner"|
|a_way|float64|cldf|Frequently used term "the wall"|
|the_payment|float64|cldf|Frequently used term "the payment"|
|the_back|float64|cldf|Frequently used term "the back"|
|the_interest|float64|cldf|Frequently used term "the interest"|
|my_payment|float64|cldf|Frequently used term "my payment"|
|the_amount|float64|cldf|Frequently used term "the amount"|
|the_area|float64|cldf|Frequently used term "the area"|
|my_credit|float64|cldf|Frequently used term "my credit"|
|some_advice|float64|cldf|Frequently used term "some advice"|
|some_sort|float64|cldf|Frequently used term "some sort"|
|the_contractor|float64|cldf|Frequently used term "the contractor"|
|an_issue|float64|cldf|Frequently used term "an issue"|
|a_couple|float64|cldf|Frequently used term "a couple"|
|any_tip|float64|cldf|Frequently used term "any tip"|
|the_outside|float64|cldf|Frequently used term "the outside"|
|the_switch|float64|cldf|Frequently used term "the switch"|
|my_credit_score|float64|cldf|Frequently used term "my credit score"|
|my_dad|float64|cldf|Frequently used term "my dad"|
|the_ground|float64|cldf|Frequently used term "the ground"|
|the_home|float64|cldf|Frequently used term "the home"|
|the_job|float64|cldf|Frequently used term "the job"|
|the_middle|float64|cldf|Frequently used term "the middle"|
|my_situation|float64|cldf|Frequently used term "my situation"|
|the_sink|float64|cldf|Frequently used term "the sink"|
|my_job|float64|cldf|Frequently used term "my job"|
|the_deck|float64|cldf|Frequently used term "the deck"|
|a_year|float64|cldf|Frequently used term "a year"|
|my_plan|float64|cldf|Frequently used term "my plan"|
|the_pipe|float64|cldf|Frequently used term "the pipe"|
|the_light|float64|cldf|Frequently used term "the light"|
|the_fence|float64|cldf|Frequently used term "the fence"|
|a_bunch|float64|cldf|Frequently used term "a bunch"|
|the_tub|float64|cldf|Frequently used term "the tub"|
|the_line|float64|cldf|Frequently used term "the line"|
|my_mom|float64|cldf|Frequently used term "my mom"|
|the_edge|float64|cldf|Frequently used term "the edge"|
|the_gap|float64|cldf|Frequently used term "the gap"|
|my_life|float64|cldf|Frequently used term "my life"|
|the_summer|float64|cldf|Frequently used term "the summer"|
|a_private_loan|float64|cldf|Frequently used term "a private loan"|
|a_cosigner|float64|cldf|Frequently used term "a cosigner"|
|any_experience|float64|cldf|Frequently used term "any experience"|
|the_future|float64|cldf|Frequently used term "the future"|
|the_title|float64|cldf|Frequently used term "the title"|
|my_family|float64|cldf|Frequently used term "my family"|
|the_company|float64|cldf|Frequently used term "the company"|
|any_insight|float64|cldf|Frequently used term "any insights"|
|the_bedroom|float64|cldf|Frequently used term "the bedroom"|
|the_stud|float64|cldf|Frequently used term "the stud"|
|my_debt|float64|cldf|Frequently used term "my debt"|
|a_picture|float64|cldf|Frequently used term "a picture"|
|the_post|float64|cldf|Frequently used term "the post"|
|the_program|float64|cldf|Frequently used term "the program"|
|the_foundation|float64|cldf|Frequently used term "the foundation"|
|these_loan|float64|cldf|Frequently used term "these loan"|
|the_option|float64|cldf|Frequently used term "the option"|
|this_sub|float64|cldf|Frequently used term "this sub"|
|the_beginning|float64|cldf|Frequently used term "the beginning"|
|the_frame|float64|cldf|Frequently used term "the frame"|
|the_corner|float64|cldf|Frequently used term "the corner"|
|some_kind|float64|cldf|Frequently used term "some kind"|
|my_account|float64|cldf|Frequently used term "my account"|
|hi_everyone|float64|cldf|Frequently used term "hi everyone"|
|a_home|float64|cldf|Frequently used term "a home"|
|a_ton|float64|cldf|Frequently used term "a ton"|
|the_cabinet|float64|cldf|Frequently used term "the cabinet"|
|the_fall|float64|cldf|Frequently used term "the fall"|
|my_home|float64|cldf|Frequently used term "my home"|
|the_space|float64|cldf|Frequently used term "the space"|
|the_paint|float64|cldf|Frequently used term "the paint"|
|the_interest_rate|float64|cldf|Frequently used term "the interest rate"|
|financial_aid|float64|cldf|Frequently used term "financial aid"|
|a_time|float64|cldf|Frequently used term "a time"|
|this_situation|float64|cldf|Frequently used term "this situation"|
|my_monthly_payment|float64|cldf|Frequently used term "my monthly payment"|
|the_outlet|float64|cldf|Frequently used term "the outlet"|
|a_payment|float64|cldf|Frequently used term "a payment"|
|my_option|float64|cldf|Frequently used term "my option"|
|my_income|float64|cldf|Frequently used term "my income"|
|2_year|float64|cldf|Frequently used term "2 year"|
|the_government|float64|cldf|Frequently used term "the government"|
|the_smell|float64|cldf|Frequently used term "the smell"|
|what_kind|float64|cldf|Frequently used term "what kind"|
|the_other_side|float64|cldf|Frequently used term "the other side"|
|the_front|float64|cldf|Frequently used term "the front"|
|the_breaker|float64|cldf|Frequently used term "the breaker"|
|the_inside|float64|cldf|Frequently used term "the inside"|

### Final conclusion/recommendations

* The most important elements are the frequently used terms and words
* Terms and words that were indicative of posts originating in r/StudentLoans were usually related to finances, college, parents, and school.
* Terms and words that were indicative of posts originating in r/HomeImprovements were usually related to building materials and portions of a house such as rooms in the house or "side", "front", "back", etc.
* Possesive references to home or house such as "my house", "our house", etc were strongly indicative of posts originating in r/HomeImprovement. However, more vague references "a house" and "a home" were actually pretty evenly balanced between the two subreddits with "a home" actually being more common in the r/StudentLoans subreddit. This tends to support the notion that people with student loans are in fact considering home ownership.
* It may be assumed that references to a spouse may be more common in the r/HomeImprovement subreddit as those posting in that subreddit may be assumed to be older. However, references to a spouse were actually almost an even split.
* Parts of speech analysis showed that "Proper Nouns" were the main part of speech indicating that a subreddit originated in the r/StudentLoans subreddit. The most indicative parts of speech for origination in the r/HomeImprovement subreddit are "Determinants" and "Other". While "Other" is a relatively vague part of speech to analyze while reading through, it was almost 6 times as likely to occur in the r/HomeImprovement subreddit so it is a good candidate for Machine Learning Purposes.
* Both average word count and average sentiment analysis score were found to be higher in the r/StudentLoans with 99% confidence, they do not really make for good comparisons to other subreddits or texts on other forums.

### References
1. Home Improvement Subreddit - https://www.reddit.com/r/HomeImprovement/new/
2. Student Loan Subreddit - https://www.reddit.com/r/StudentLoans/new/
3. Student Loan Debt and Housing Report 2017 - https://cdn.nar.realtor/sites/default/files/documents/2017-student-loan-debt-and-housing-09-26-2017.pdf
4. Home Improvement Survey - https://www.bankrate.com/loans/personal-loans/coronavirus-loans-home-improvement-survey/
5. Ignore Warnings = Peter Yonka, multiple warnings during GridSearch phase
6. Spacy Parts of Speach Guide - https://ashutoshtripathi.com/2020/04/13/parts-of-speech-tagging-and-dependency-parsing-using-spacy-nlp/
7. Average Credit Card Rates - https://www.thebalance.com/average-credit-card-interest-rate-february-2021-5120542#:~:text=The%20average%20credit%20card%20interest%20rate%20was%20a,purchase%20or%20balance%20transfer%20interest-rate%20changes%20in%20February.
8. Home Equity Rates - https://www.valuepenguin.com/mortgages/average-home-equity-loan-rates