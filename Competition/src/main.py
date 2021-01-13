import sys
import subprocess
import time
from pathlib import Path
import xml.etree.ElementTree as ElementTree
import wandb


indri_install_dir = Path('../indri-5.8-install')
trec_eval_dir = Path('../trec_eval_9.0')

run_query_bin = indri_install_dir / 'bin/IndriRunQuery'
trec_eval_bin = trec_eval_dir / 'trec_eval'

qrels_file = sys.argv[1]  # Path to qrels
query_file = sys.argv[2]  # Path to queries


def read_queries(path: Path):
    root = ElementTree.parse(path).getroot()
    queries = []
    for query in root:
        number = query[0].text
        text = query[1].text
        # Get rid of indri query language since its just #combine anyway
        if text.startswith('#combine( ') and text.endswith(' )'):
            text = text[len('#combine( '):-len(' )')]
        queries.append((number, text))
    return queries


def write_queries(queries, path):
    with path.open('w') as f:
        f.write('<parameters>\n')
        for query in queries:
            f.write('\t<query>\n')
            f.write(f'\t\t<number>{query[0]}</number>\n')
            f.write(f'\t\t<text>{query[1]}</text>\n')
            f.write('\t</query>\n')
        f.write('</parameters>')


def parse_results(results_text):
    results = {}
    for result in results_text.splitlines():
        query, _, doc, _, score, _ = result.split()
        if query not in results:
            results[query] = []
        results[query].append((doc, score))
    return results


def write_results(path, results, run_tag='indri'):
    with path.open('w') as f:
        for query, rankings in results.items():
            for rank, (doc, score) in enumerate(rankings):
                f.write(f'{query} Q0 {doc} {rank + 1} {score} {run_tag}\n')


def make_parameters(method):
    stopwords = ''
    with Path('stopwords.txt').open('r') as f:
        for word in f.read().splitlines():
            stopwords += f'<word>{word}</word>\n'
    parameters = f'<parameters>\n<memory>1G</memory>\n<index>/data/IRCompetition/ROBUSTindex</index>\n' \
                 f'<count>1000</count>\n<trecFormat>true</trecFormat>\n{method}\n' \
                 f'<stopper>\n{stopwords}</stopper>\n</parameters>'
    return parameters


def run_query(parameters, queries):
    t = time.time()  # TODO use temporary files instead of this to allow parallel runs
    tmp_param_file = Path(f'qparams_{t}.tmp')
    tmp_query_file = Path(f'queries_{t}.tmp')
    with tmp_param_file.open('w') as p:
        p.write(parameters)
    write_queries(queries, tmp_query_file)
    result = subprocess.check_output([run_query_bin, tmp_param_file, tmp_query_file])
    tmp_param_file.unlink()
    tmp_query_file.unlink()
    return parse_results(result.decode())


def trec_eval(qrels, results):
    t = time.time()  # TODO use temporary files instead of this to allow parallel runs
    tmp_qrels_file = Path(f'qrels_{t}.tmp')
    tmp_result_file = Path(f'result_{t}.tmp')
    with tmp_qrels_file.open('w') as f:
        f.write(qrels)
    write_results(tmp_result_file, results)
    out = subprocess.check_output([trec_eval_bin, '-m', 'map', tmp_qrels_file, tmp_result_file])
    tmp_qrels_file.unlink()
    tmp_result_file.unlink()
    return float(out.split()[-1])  # map


def eval_queries(query_parameters, queries):
    qrels = ''
    with Path(qrels_file).open('r') as r:
        for line in r.read().splitlines():
            number = line.split()[0]
            if number in [n for n, _ in queries]:
                qrels += f'{line}\n'
    results = run_query(query_parameters, queries)
    map_score = trec_eval(qrels, results)
    return map_score


def eval_method(method, folds=5, wandb_log=False):
    # folds=0 means evaluate on all queries
    parameters = make_parameters(method)
    queries = read_queries(Path(query_file))
    if folds:
        avg_map = 0
        fold_size = int(len(queries) / folds)
        for fold in range(folds):
            avg_map += eval_queries(parameters, queries[fold * fold_size:(fold + 1) * fold_size])
        avg_map /= folds
        if wandb_log:
            wandb.log({'MAP': avg_map})
    else:
        total_map = eval_queries(parameters, queries)
        if wandb_log:
            wandb.log({'MAP': total_map})
    # models = Path(f'models_{qrels_file}')
    # models.mkdir(parents=True, exist_ok=True)
    # with (models / f'model_{avg_map if folds else total_map:.4f}.txt').open('w') as f:
    #     f.write(method)


def dirichlet():
    wandb.init(project='ir_competition')
    print(f'dir mu={wandb.config.mu}')
    eval_method(f'<rule>method:dir,mu:{wandb.config.mu}</rule>'
                f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms>'
                f'<fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>',
                wandb_log=True)


def jm():
    wandb.init(project='ir_competition')
    print(f'jm collectionLambda={wandb.config.collectionLambda},documentLambda={wandb.config.documentLambda}')
    eval_method(f'<rule>method:jm,collectionLambda:{wandb.config.collectionLambda},'
                f'documentLambda:{wandb.config.documentLambda}</rule>'
                f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms>'
                f'<fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>',
                wandb_log=True)


def two():
    wandb.init(project='ir_competition')
    print(f'two mu={wandb.config.mu},lambda={wandb.config["lambda"]}')
    eval_method(f'<rule>method:two,mu:{wandb.config.mu},lambda:{wandb.config["lambda"]}</rule>'
                f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms>'
                f'<fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>',
                wandb_log=True)


def tfidf():
    wandb.init(project='ir_competition')
    print(f'tfidf k1={wandb.config.k1},b={wandb.config.b}')
    eval_method(f'<baseline>tfidf,k1:{wandb.config.k1},b:{wandb.config.b}</baseline>'
                f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms>'
                f'<fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>',
                wandb_log=True)


def okapi():
    wandb.init(project='ir_competition')
    print(f'okapi k1={wandb.config.k1},b={wandb.config.b},k3={wandb.config.k3}')
    eval_method(f'<baseline>okapi,k1:{wandb.config.k1},b:{wandb.config.b},k3:{wandb.config.k3}</baseline>'
                f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms>'
                f'<fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>',
                wandb_log=True)


def make_weighted_copeland(ranking_lists, folds=5):
    def weighted_copeland():
        wandb.init(project='ir_competition')
        weights = [wandb.config.dir, wandb.config.jm, wandb.config.two, wandb.config.tfidf, wandb.config.okapi]
        query_rankings = {}
        for results in ranking_lists:
            for query, rankings in results.items():
                if query not in query_rankings:
                    query_rankings[query] = []
                query_rankings[query].append([doc for doc, _ in rankings])  # Current ranking list
        fused = {}
        for query, rankings in query_rankings.items():
            fused[query] = copeland(rankings, weights)
        with Path(qrels_file).open('r') as r:
            qrels = r.read()
        avg_map = 0
        fold_size = int(len(fused) / folds)
        for fold in range(folds):
            queries_fold = list(fused)[fold * fold_size:(fold + 1) * fold_size]
            qrels_fold = ''
            for line in qrels.splitlines():
                number = line.split()[0]
                if number in queries_fold:
                    qrels_fold += f'{line}\n'
            avg_map += trec_eval(qrels_fold, {query: rank for query, rank in fused.items() if query in queries_fold})
        avg_map /= folds
        wandb.log({'MAP': avg_map})
        print(avg_map)
    return weighted_copeland


def make_sweeps():
    wandb.sweep({
        'name': 'dir',
        'method': 'random',
        'metric': {
            'name': 'MAP',
            'goal': 'maximize'
        },
        'parameters': {
            'mu': {
                'distribution': 'int_uniform',
                'min': 100,
                'max': 3000
            },
            'fbDocs': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 15
            },
            'fbTerms': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 15
            },
            'fbMu': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 150
            },
            'fbOrigWeight': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            }
        }
    }, project='ir_competition')
    wandb.sweep({
        'name': 'jm',
        'method': 'random',
        'metric': {
            'name': 'MAP',
            'goal': 'maximize'
        },
        'parameters': {
            'collectionLambda': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            },
            'documentLambda': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            },
            'fbDocs': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 15
            },
            'fbTerms': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 15
            },
            'fbMu': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 150
            },
            'fbOrigWeight': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            }
        }
    }, project='ir_competition')
    wandb.sweep({
        'name': 'two',
        'method': 'random',
        'metric': {
            'name': 'MAP',
            'goal': 'maximize'
        },
        'parameters': {
            'mu': {
                'distribution': 'int_uniform',
                'min': 100,
                'max': 3000
            },
            'lambda': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            },
            'fbDocs': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 15
            },
            'fbTerms': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 15
            },
            'fbMu': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 150
            },
            'fbOrigWeight': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            }
        }
    }, project='ir_competition')
    wandb.sweep({
        'name': 'tfidf',
        'method': 'random',
        'metric': {
            'name': 'MAP',
            'goal': 'maximize'
        },
        'parameters': {
            'k1': {
                'distribution': 'uniform',
                'min': 0,
                'max': 3
            },
            'b': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            },
            'fbDocs': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 15
            },
            'fbTerms': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 15
            },
            'fbMu': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 150
            },
            'fbOrigWeight': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            }
        }
    }, project='ir_competition')
    wandb.sweep({
        'name': 'okapi',
        'method': 'random',
        'metric': {
            'name': 'MAP',
            'goal': 'maximize'
        },
        'parameters': {
            'k1': {
                'distribution': 'uniform',
                'min': 0,
                'max': 3
            },
            'b': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            },
            'k3': {
                'distribution': 'int_uniform',
                'min': 4,
                'max': 10
            },
            'fbDocs': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 15
            },
            'fbTerms': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 15
            },
            'fbMu': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 150
            },
            'fbOrigWeight': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            }
        }
    }, project='ir_competition')
    wandb.sweep({
        'name': 'weighted_copeland',
        'method': 'random',
        'metric': {
            'name': 'MAP',
            'goal': 'maximize'
        },
        'parameters': {
            'dir': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            },
            'jm': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            },
            'two': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            },
            'tfidf': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            },
            'okapi': {
                'distribution': 'uniform',
                'min': 0,
                'max': 1
            }
        }
    }, project='ir_competition')


def sweep(sweep_id, function):
    wandb.agent(sweep_id, function=function, project='ir_competition')


def borda(rankings):
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += len(ranking) - 1 - rank
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]


def copeland(rankings, weights=None):
    pairwise = {}
    documents = set()
    scores = {}
    for m, ranking in enumerate(rankings):
        for rank, doc in enumerate(ranking):
            documents.add(doc)
            for rank2, doc2 in enumerate(ranking):
                if rank2 <= rank:
                    continue
                if (doc, doc2) not in pairwise:
                    pairwise[doc, doc2] = 0
                    pairwise[doc2, doc] = 0
                pairwise[doc, doc2] += weights[m] if weights else 1
                pairwise[doc2, doc] -= weights[m] if weights else 1
    for doc in documents:
        if doc not in scores:
            scores[doc] = 0
        for doc2 in documents:
            if doc == doc2:
                continue
            if (doc, doc2) in pairwise and pairwise[doc, doc2] > 0:
                scores[doc] += 1
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]


def fusion(ranking_lists, copeland_weights):
    query_rankings = {}
    for results in ranking_lists:
        for query, rankings in results.items():
            if query not in query_rankings:
                query_rankings[query] = []
            query_rankings[query].append([doc for doc, _ in rankings])  # Current ranking list
    fused_borda = {}
    fused_copeland = {}
    fused_weighted_copeland = {}
    for query, rankings in query_rankings.items():
        fused_borda[query] = borda(rankings)
        fused_copeland[query] = copeland(rankings)
        fused_weighted_copeland[query] = copeland(rankings, copeland_weights)
    return fused_borda, fused_copeland, fused_weighted_copeland


def main(evaluate=False):
    # make_sweeps()  # Run this only once then copy sweep ids
    # sweep('6nx1hhp2', dirichlet)
    # sweep('o0qo77i2', jm)
    # sweep('bfeu8caz', two)
    # sweep('wc1042df', tfidf)
    # sweep('vxms7eoe', okapi)
    best_models = ['<rule>method:dir,mu:990</rule>'  # dir
                   '<fbDocs>14</fbDocs><fbTerms>12</fbTerms><fbMu>71</fbMu><fbOrigWeight>0.4308</fbOrigWeight>',
                   '<rule>method:jm,collectionLambda:0.2534,documentLambda:0.8633</rule>'  # jm
                   '<fbDocs>12</fbDocs><fbTerms>11</fbTerms><fbMu>32</fbMu><fbOrigWeight>0.3892</fbOrigWeight>',
                   '<rule>method:two,mu:325,lambda:0.4762</rule>'  # two
                   '<fbDocs>15</fbDocs><fbTerms>13</fbTerms><fbMu>103</fbMu><fbOrigWeight>0.4352</fbOrigWeight>',
                   '<baseline>tfidf,k1:0.4844,b:0.4867</baseline>'  # tfidf
                   '<fbDocs>10</fbDocs><fbTerms>10</fbTerms><fbMu>77</fbMu><fbOrigWeight>0.8694</fbOrigWeight>',
                   '<baseline>okapi,k1:0.3899,b:0.76,k3:9</baseline>'  # okapi
                   '<fbDocs>8</fbDocs><fbTerms>10</fbTerms><fbMu>63</fbMu><fbOrigWeight>0.7273</fbOrigWeight>']
    ranking_lists = []
    queries = read_queries(Path(query_file))
    for model in best_models:
        ranking_lists.append(run_query(make_parameters(model), queries))
    # sweep('rju05a6n', make_weighted_copeland(ranking_lists))
    fusion_results = fusion(ranking_lists, copeland_weights=[0.4201, 0.2555, 0.8505, 0.6863, 0.6547])
    fused_borda, fused_copeland, fused_weighted_copeland = fusion_results
    if evaluate:
        with Path(qrels_file).open('r') as r:
            qrels = r.read()
            print('ranking lists', [trec_eval(qrels, rankings) for rankings in ranking_lists])
            print('borda', trec_eval(qrels, fused_borda))
            print('copeland', trec_eval(qrels, fused_copeland))
            print('copeland_weighted', trec_eval(qrels, fused_weighted_copeland))
    write_results(Path('run_1.res'), fused_borda, run_tag='run1')
    write_results(Path('run_2.res'), fused_copeland, run_tag='run2')
    write_results(Path('run_3.res'), fused_weighted_copeland, run_tag='run3')


if __name__ == '__main__':
    main(evaluate=True)
