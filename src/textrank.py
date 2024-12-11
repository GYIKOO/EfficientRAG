import editdistance
import itertools
import networkx as nx
import nltk
import argparse
import json
from tqdm import tqdm
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fpath", type=str,
    ),
    parser.add_argument(
        "--top_k", type=int,
    )
    args = parser.parse_args()
    return args


def build_graph(nodes):
    """Return a networkx graph instance.

    :param nodes: List of hashables that represent the nodes of a graph.
    """
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        levDistance = editdistance.eval(firstString, secondString)
        gr.add_edge(firstString, secondString, weight=levDistance)

    return gr


def extract_articles(articles,top_k):
    graph = build_graph(articles)
    calculated_page_rank = nx.pagerank(graph, weight='weight')
    # most important sentences in ascending order of importance
    articles = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)
    return articles[:top_k]

def extract_sentences(text, summary_length=100, clean_sentences=False, language='english'):
    """Return a paragraph formatted summary of the source text.

    :param text: A string.
    """
    sent_detector = nltk.data.load('tokenizers/punkt/'+language+'.pickle')
    sentence_tokens = sent_detector.tokenize(text.strip())
    graph = build_graph(sentence_tokens)

    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)

    # return a 100 word summary
    summary = ' '.join(sentences)
    summary_words = summary.split()
    summary_words = summary_words[0:summary_length]
    dot_indices = [idx for idx, word in enumerate(summary_words) if word.find('.') != -1]
    if clean_sentences and dot_indices:
        last_dot = max(dot_indices) + 1
        summary = ' '.join(summary_words[0:last_dot])
    else:
        summary = ' '.join(summary_words)

    return summary


def read_jsonl(filepath):
    data = []
    with open(filepath,"r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
        return data
    
def write_jsonl(outpath, data):
    with jsonlines.open(outpath,mode="w") as writer:
        for datum in data:
            writer.write(datum)
        print("write to "+outpath)

def filter_chunks(fpath,top_k=2):
    origin_chunks_count, filtered_chunks_count = 0,0
    outpath = fpath.replace(".jsonl",f"filtered_top{top_k}.jsonl")
    POS_TAG, NEG_TAG = "<CONTINUE>", "<TERMINATE>"
    # load file with retrieved chunks
    data = read_jsonl(fpath)
    # filter!
    for i,datum in enumerate(tqdm(data)):
        # collect all chunks labelled as POS_TAG
        valid_docs=[]
        for hop_idx in range(4):
            if str(hop_idx) not in datum.keys():
                break
            for doc in datum[str(hop_idx)]["docs"]:
                if doc["label"] == POS_TAG:
                    valid_docs.append(doc["text"])
        # select top_k important chunks
        top_k = max(1,top_k)
        # label others as NEG_TAG
        keep_docs = extract_articles(valid_docs,top_k)
        delete_docs = [x for x in valid_docs if x not in keep_docs]
        origin_chunks_count += len(valid_docs)
        filtered_chunks_count += len(keep_docs)
        for hop_idx in range(4):
            if str(hop_idx) not in datum.keys():
                break
            for j,doc in enumerate(datum[str(hop_idx)]["docs"]):
                if doc["text"] in delete_docs:
                    try:
                        data[i][str(hop_idx)]["docs"][j]["label"] = NEG_TAG
                    except:
                        breakpoint()
    write_jsonl(outpath,data)
    print(f"========before==========\n[SUM]{origin_chunks_count}     [AVG]{origin_chunks_count/len(data)}")
    print(f"========after==========\n[SUM]{filtered_chunks_count}     [AVG]{filtered_chunks_count/len(data)}")
    print(f"{(1-filtered_chunks_count/origin_chunks_count)*100}% chunks pruned")

        
        
            
    
    


if __name__ == "__main__":
    args = parse_args()

    filter_chunks(args.fpath,args.top_k)
