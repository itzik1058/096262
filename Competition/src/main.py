import sys
import subprocess
import time
from pathlib import Path
import xml.etree.ElementTree as ElementTree
import wandb


qrels_file = sys.argv[1]
query_file = sys.argv[2]


def read_queries(path: Path):
    root = ElementTree.parse(path).getroot()
    queries = []
    for query in root:
        number = query[0].text
        words = query[1].text
        queries.append((number, words))
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


def eval(method, folds=5):
    stopwords = ''
    with Path('stopwords.txt').open('r') as f:
        for word in f.read().splitlines():
            stopwords += f'<word>{word}</word>\n'
    run = time.time()  # Allows multiple runs
    with Path(f'indriRunQuery_{run}.xml').open('w') as f:
        f.write(f'<parameters>\n<memory>1G</memory>\n<index>/data/IRCompetition/ROBUSTindex</index>\n<count>1000</count>\n'
                f'<trecFormat>true</trecFormat>\n{method}\n<stopper>\n{stopwords}</stopper>\n</parameters>')
    avg_map = 0
    queries = read_queries(Path(query_file))
    fold_size = int(len(queries) / folds)
    for fold in range(folds):
        write_queries(queries[fold * fold_size:(fold + 1) * fold_size], Path(f'val_{run}.queries'))
        result = subprocess.check_output(['../indri-5.8-install/bin/IndriRunQuery', f'indriRunQuery_{run}.xml', f'val_{run}.queries'])
        Path(f'val_{run}.queries').unlink()
        with Path(qrels_file).open('r') as r:
            with Path(f'val_{run}.qrels').open('w') as w:
                for line in r.read().splitlines():
                    number = line.split()[0]
                    if number in [n for n, _ in queries[fold * fold_size:(fold + 1) * fold_size]]:
                        w.write(line + '\n')
        with Path(f'result_{run}').open('w') as f:
            f.write(result.decode())
        map = float(subprocess.check_output(['../trec_eval_9.0/trec_eval', '-m', 'map', f'val_{run}.qrels',
                                             f'result_{run}']).split()[-1])
        avg_map += map
        Path(f'val_{run}.qrels').unlink()
        Path(f'result_{run}').unlink()
    avg_map /= folds
    Path(f'indriRunQuery_{run}.xml').unlink()
    wandb.log({'MAP': avg_map})
    models = Path(f'models_{qrels_file}')
    models.mkdir(parents=True, exist_ok=True)
    with (models / f'model_{avg_map:.4f}.txt').open('w') as f:
        f.write(method)


def dir():
    wandb.init(project='ir_competition')
    print(f'dir mu={wandb.config.mu}')
    eval(f'<rule>method:dir,mu:{wandb.config.mu}</rule>'
         f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms><fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>')


def jm():
    wandb.init(project='ir_competition')
    print(f'jm collectionLambda={wandb.config.collectionLambda},documentLambda={wandb.config.documentLambda}')
    eval(f'<rule>method:jm,collectionLambda:{wandb.config.collectionLambda},documentLambda:{wandb.config.documentLambda}</rule>'
         f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms><fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>')


def two():
    wandb.init(project='ir_competition')
    print(f'two mu={wandb.config.mu},lambda={wandb.config["lambda"]}')
    eval(f'<rule>method:two,mu:{wandb.config.mu},lambda:{wandb.config["lambda"]}</rule>'
         f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms><fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>')


def tfidf():
    wandb.init(project='ir_competition')
    print(f'tfidf k1={wandb.config.k1},b={wandb.config.b}')
    eval(f'<baseline>tfidf,k1:{wandb.config.k1},b:{wandb.config.b}</baseline>'
         f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms><fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>')


def okapi():
    wandb.init(project='ir_competition')
    print(f'okapi k1={wandb.config.k1},b={wandb.config.b},k3={wandb.config.k3}')
    eval(f'<baseline>okapi,k1:{wandb.config.k1},b:{wandb.config.b},k3:{wandb.config.k3}</baseline>'
         f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms><fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>')


def rm():
    print(f'rm fbDocs={wandb.config.fbDocs},fbTerms={wandb.config.fbTerms},fbMu={wandb.config.fbMu},fbOrigWeight={wandb.config.fbOrigWeight}')
    eval(f'<baseline>okapi,k1:0.8,b:0.3,k3:9</baseline>'
         f'<fbDocs>{wandb.config.fbDocs}</fbDocs><fbTerms>{wandb.config.fbTerms}</fbTerms><fbMu>{wandb.config.fbMu}</fbMu><fbOrigWeight>{wandb.config.fbOrigWeight}</fbOrigWeight>')


if __name__ == '__main__':
    # dir_sweep = wandb.sweep({
    #     'name': 'dir',
    #     'method': 'random',
    #     'metric': {
    #         'name': 'MAP',
    #         'goal': 'maximize'
    #     },
    #     'parameters': {
    #         'mu': {
    #             'distribution': 'int_uniform',
    #             'min': 100,
    #             'max': 3000
    #         },
    #         'fbDocs': {
    #             'distribution': 'int_uniform',
    #             'min': 1,
    #             'max': 15
    #         },
    #         'fbTerms': {
    #             'distribution': 'int_uniform',
    #             'min': 1,
    #             'max': 15
    #         },
    #         'fbMu': {
    #             'distribution': 'int_uniform',
    #             'min': 5,
    #             'max': 150
    #         },
    #         'fbOrigWeight': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 1
    #         }
    #     }
    # }, project='ir_competition')
    # jm_sweep = wandb.sweep({
    #     'name': 'jm',
    #     'method': 'random',
    #     'metric': {
    #         'name': 'MAP',
    #         'goal': 'maximize'
    #     },
    #     'parameters': {
    #         'collectionLambda': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 1
    #         },
    #         'documentLambda': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 1
    #         },
    #         'fbDocs': {
    #             'distribution': 'int_uniform',
    #             'min': 1,
    #             'max': 15
    #         },
    #         'fbTerms': {
    #             'distribution': 'int_uniform',
    #             'min': 1,
    #             'max': 15
    #         },
    #         'fbMu': {
    #             'distribution': 'int_uniform',
    #             'min': 5,
    #             'max': 150
    #         },
    #         'fbOrigWeight': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 1
    #         }
    #     }
    # }, project='ir_competition')
    # two_sweep = wandb.sweep({
    #     'name': 'two',
    #     'method': 'random',
    #     'metric': {
    #         'name': 'MAP',
    #         'goal': 'maximize'
    #     },
    #     'parameters': {
    #         'mu': {
    #             'distribution': 'int_uniform',
    #             'min': 100,
    #             'max': 3000
    #         },
    #         'lambda': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 1
    #         },
    #         'fbDocs': {
    #             'distribution': 'int_uniform',
    #             'min': 1,
    #             'max': 15
    #         },
    #         'fbTerms': {
    #             'distribution': 'int_uniform',
    #             'min': 1,
    #             'max': 15
    #         },
    #         'fbMu': {
    #             'distribution': 'int_uniform',
    #             'min': 5,
    #             'max': 150
    #         },
    #         'fbOrigWeight': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 1
    #         }
    #     }
    # }, project='ir_competition')
    # tfidf_sweep = wandb.sweep({
    #     'name': 'tfidf',
    #     'method': 'random',
    #     'metric': {
    #         'name': 'MAP',
    #         'goal': 'maximize'
    #     },
    #     'parameters': {
    #         'k1': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 3
    #         },
    #         'b': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 1
    #         },
    #         'fbDocs': {
    #             'distribution': 'int_uniform',
    #             'min': 1,
    #             'max': 15
    #         },
    #         'fbTerms': {
    #             'distribution': 'int_uniform',
    #             'min': 1,
    #             'max': 15
    #         },
    #         'fbMu': {
    #             'distribution': 'int_uniform',
    #             'min': 5,
    #             'max': 150
    #         },
    #         'fbOrigWeight': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 1
    #         }
    #     }
    # }, project='ir_competition')
    # okapi_sweep = wandb.sweep({
    #     'name': 'okapi',
    #     'method': 'random',
    #     'metric': {
    #         'name': 'MAP',
    #         'goal': 'maximize'
    #     },
    #     'parameters': {
    #         'k1': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 3
    #         },
    #         'b': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 1
    #         },
    #         'k3': {
    #             'distribution': 'int_uniform',
    #             'min': 4,
    #             'max': 10
    #         },
    #         'fbDocs': {
    #             'distribution': 'int_uniform',
    #             'min': 1,
    #             'max': 15
    #         },
    #         'fbTerms': {
    #             'distribution': 'int_uniform',
    #             'min': 1,
    #             'max': 15
    #         },
    #         'fbMu': {
    #             'distribution': 'int_uniform',
    #             'min': 5,
    #             'max': 150
    #         },
    #         'fbOrigWeight': {
    #             'distribution': 'uniform',
    #             'min': 0,
    #             'max': 1
    #         }
    #     }
    # }, project='ir_competition')
    dir_sweep = '6nx1hhp2'
    jm_sweep = 'o0qo77i2'
    two_sweep = 'bfeu8caz'
    tfidf_sweep = 'wc1042df'
    okapi_sweep = 'vxms7eoe'
    wandb.agent(dir_sweep, function=dir, project='ir_competition')
    # wandb.agent(jm_sweep, function=jm, project='ir_competition')
    # wandb.agent(two_sweep, function=two, project='ir_competition')
    # wandb.agent(tfidf_sweep, function=tfidf, project='ir_competition')
    # wandb.agent(okapi_sweep, function=okapi, project='ir_competition')
