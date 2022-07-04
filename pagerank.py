import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Get number of pages in corpus
    total_pages = len(corpus)
    # Get number of links in page
    num_links = len(corpus[page])
    # If no links on page
    if num_links == 0:
        # Get chance to navigate to any page
        chance = 1.0 / total_pages
        # Populate probability distribution with chance
        dist = dict.fromkeys(list(corpus.keys()), chance)
        # Return probability distribution
        return dist
    else:
        # Get random chance to navigate to any page in the corpus
        random_chance = (1.0 - damping_factor) / total_pages
        # Get chance to navigate to one of the links on page
        link_chance = damping_factor / num_links
        # Populate probability distribution with random_chance
        dist = dict.fromkeys(list(corpus.keys()), random_chance)
        # Add link_chance to linked pages
        for linked_page in corpus[page]:
            dist[linked_page] += link_chance
        # Return probability distribution
        return dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Get random starting page
    page = random.choice(list(corpus.keys()))
    # Get pages probability distribution
    dist = transition_model(corpus, page, damping_factor)
    # Initialise sample dictionary with keys from corpus and values of 0
    samples = dict.fromkeys(list(corpus.keys()), 0)
    # Add page to sample dictionary
    samples[page] += 1
    # Make n - 1 samples
    for s in range(n - 1):
        # Get random page from corpus given a probability distribution
        page = random.choices(population=list(dist.keys()), weights=list(dist.values()), k=1).pop()
        # Get pages probability distribution
        dist = transition_model(corpus, page, damping_factor)
        # Add page to sample dictionary
        samples[page] += 1
    # Initialise page_rank dictionary
    page_rank = dict()
    for s in samples:
        # Get probability that page s was in the sample
        page_rank[s] = samples[s] / n
    # Return page_rank
    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialise linking_pages dictionary
    linking_pages_dict = dict.fromkeys(list(corpus))
    # Iterate over pages in corpus to find linking pages
    for page in corpus:
        # Initialise array value
        linking_pages_dict[page] = []
        for linking_page in corpus:
            # If page is linked to by linking_page or linking_page has no links
            if page in corpus[linking_page] or len(corpus[linking_page]) == 0:
                # Add linking_page to linking_pages_dict[page]
                linking_pages_dict[page].append(linking_page)
    # Get total number of pages
    total_pages = len(corpus)
    # Initialise page_rank dictionary with (1 / total_pages) values
    page_rank = dict.fromkeys(list(corpus.keys()), 1 / total_pages)
    # Estimate page_rank until there is a difference less than or equal to threshold
    loop = True
    threshold = 0.001
    while loop:
        loop = False
        # Loop all pages in corpus
        for page in corpus:
            # Get estimated page rank for page
            estimate = estimate_page_rank(corpus, damping_factor, total_pages, page_rank, linking_pages_dict[page])
            # If the difference between estimate and current page_rank[page] is greater than threshold
            current = page_rank[page]
            if current - estimate > threshold or estimate - current > threshold:
                # Change was greater than threshold, so keep looping
                loop = True
            # Set new page_rank value
            page_rank[page] = estimate
    # Return page_rank
    return page_rank


def estimate_page_rank(corpus, damping_factor, total_pages, current_page_ranks, linking_pages):
    # Get component of page_rank given current_page_rank and linking_pages
    component = 0
    for linking_page in linking_pages:
        # Get number of links on linking_page
        num_links = len(corpus[linking_page])
        # If linking_page has no links
        if num_links == 0:
            # Add estimated_page_rank divided by total_pages
            component += current_page_ranks[linking_page] / total_pages
        else:
            # Add estimated_page_rank divided by num_links
            component += current_page_ranks[linking_page] / num_links
    # Return estimated page rank value
    return (1 - damping_factor) / total_pages + damping_factor * component


if __name__ == "__main__":
    main()
