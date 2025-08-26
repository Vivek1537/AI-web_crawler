import requests
from bs4 import BeautifulSoup

url = "https://www.ycombinator.com/companies"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Example: print all links
for link in soup.find_all("a", href=True):
    print(link["href"])
