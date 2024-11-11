import argparse
import pickle
import torch
import DependencyParsing as dp
import warnings
from sklearn.exceptions import ConvergenceWarning


def get_parser():
    parser = argparse.ArgumentParser(description="Depdendency parsing")
    parser.add_argument('tree_bank_file', default=None, help="Le fichier TREE_BANK_FILE contient les arbres de toutes les phrases du corpus au format conllu.")
    parser.add_argument('train_file', default=None, help="Le fichier TRAIN_FILE contient les id des phrases pour le train set.")
    parser.add_argument('dev_file', default=None, help="Le fichier DEV_FILE contient les id des phrases pour le dev set.")
    parser.add_argument('test_file', default=None, help="Le fichier TEST_FILE contient les id des phrases pour le test set.")
    parser.add_argument('arc_mode', choices=['standard', 'eager'], help="Mode d'arc utilisé pour le training oracle.")
    parser.add_argument('classifier_type', choices=['perceptron', 'mlp'], help="Type de classifieur utilisé pour le dependancy parsing: un perceptron ou un mlp.")
    parser.add_argument('-nb_epochs', default=5, help="Le nombre d'epochs pour l'entrainement.")
    parser.add_argument('-sentence_id', default="annodis.er_00001", help="Une id de phrase dont l'arbre de dépendance sera affiché.")
    parser.add_argument('-l', "--labeled", action="store_true", help="Uses labeled arcs instead of simple arcs. Default=False")
    parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
    return parser


def get_classifier(params, corpus, arc_mode, BATCH_SIZE = 25, lr=0.0005, HIDDEN_LAYER_SIZE=150, CONTEXT_SIZE=2):
    if params.classifier_type == "perceptron":
        return dp.P(corpus, arc_mode)
    if params.classifier_type == "mlp":
        return dp.MLP(corpus, arc_mode, BATCH_SIZE=BATCH_SIZE, lr=lr, HIDDEN_LAYER_SIZE=HIDDEN_LAYER_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
    return dp.Classifier()


if __name__ == '__main__':
    # ignore warnings of the perceptron :)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    # parser parameters
    parser = get_parser()
    params = parser.parse_args()

    # read file and create corpus
    corpus = dp.Corpus(params.tree_bank_file, params.train_file, params.test_file, params.dev_file)
    corpus.read_file()

    # hyperparameters
    lr = 0.000025
    hidden_layer_size = 400
    batch_size = 50
    context_size = 2
    use_labels = params.labeled
    nb_epochs = params.nb_epochs

    # arc mode
    arc_mode = corpus.create_arc_mode(use_labels, params.arc_mode)

    # Create the classifier
    transition_based_dependency_parser = get_classifier(params, corpus, arc_mode, BATCH_SIZE=batch_size, lr=lr, HIDDEN_LAYER_SIZE=hidden_layer_size, CONTEXT_SIZE=context_size)

    # TRAIN THE CLASSIFIER
    transition_based_dependency_parser.train(nb_epochs)

    # EVALUATE THE CLASSIFIER
    n=10

    transition_based_dependency_parser.evaluate('train')
    transition_based_dependency_parser.error_analysis_transition('train', n)
    transition_based_dependency_parser.evaluate('dev')
    transition_based_dependency_parser.error_analysis_transition('dev', n)
    transition_based_dependency_parser.evaluate('test')
    transition_based_dependency_parser.error_analysis_transition('test', n)

    transition_based_dependency_parser.tree_score_and_errors('train', n)
    transition_based_dependency_parser.tree_score_and_errors('dev', n)
    transition_based_dependency_parser.tree_score_and_errors('test', n)
    transition_based_dependency_parser.pretty_print_tree(params.sentence_id)
