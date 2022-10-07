from sentence_transformers import SentenceTransformer, util
import torch
from summarizer.sbert import SBertSummarizer



model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
embedder = SentenceTransformer('all-mpnet-base-v2')




"""
Пример функции: тут залетает новость, из заголовка, текста и выжимки делаем вектор
"""

news_1_header = "Рэпер Паша Техник впал в кому после вечеринки"
news_1_text = """
Российский рэп-исполнитель Паша Техник впал в кому после вечеринки. Как сообщила в социальной сети жена музыканта Ева Карицкая, во время тусовки он употреблял наркотические вещества.

«Паша в коме. Засужу всех, кто был вчера на тусовке, обещаю. Мне *** [все равно], что он сам виноват. Все пидарасы, кто употреблял с ним и звал попить кодеин, бутират, горите в аду.

Как, *** [блин], зная, что у человека началась эпилепсия и за месяц это третий раз, пить бутират, кодеин и есть ксан (ксанакс — успокаивающее средство, которое считается наркотиком. — Sport24), сука, *** [блин], орать, что он трезвый? Где??? Лучше бы в ребе (реабилитационной клинике. — Sport24) лежал. Сука.

Я жопу разорву, а не сердце, каждому, кто было вчера с ним», — написала супруга рэпера.
"""
news_1_summ = model(news_1_text, num_sentences=4)
news_1_all = embedder.encode(news_1_header, convert_to_tensor=True) + embedder.encode(news_1_text, convert_to_tensor=True) + embedder.encode(news_1_summ, convert_to_tensor=True)




"""
Тут мы замеряем косинусную близость - все, что выше 7.9 - близко и считается дублем
"""



cos_scores = util.cos_sim(news_1_all , news_6_all)[0]
