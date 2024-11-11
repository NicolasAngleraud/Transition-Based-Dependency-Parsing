import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
matplotlib.use('TkAgg')
from sklearn.metrics import f1_score
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from random import shuffle
from collections import Counter


class Arc:

    def __init__(self, dependent, head, label):
        self.dependent = dependent
        self.head = head
        self.label = label

    def __eq__(self, other):
        return self.dependent == other.dependent and self.head == other.head and self.label == other.label


class Tree:

    def __init__(self, sentence_id, list_of_arcs):
        self.sentence_id = sentence_id
        self.arcs = [Arc(arc.dependent, arc.head, arc.label) for arc in list_of_arcs]
        paths = [(arc.head, arc.dependent) for arc in self.arcs]
        length_before = 0
        while len(paths) != length_before:
            length_before = len(paths)
            for first_path in paths:
                for second_path in paths:
                    if first_path[0] == second_path[1] and not (second_path[0], first_path[1]) in paths:
                        paths.append((second_path[0], first_path[1]))
        self.paths = paths

    def is_projective_arc(self, arc):
        return all([(arc.head, word) in self.paths for word in range(min(arc.head, arc.dependent)+1, max(arc.head, arc.dependent))])

    def is_projective(self):
        return all([self.is_projective_arc(arc) for arc in self.arcs])


class Corpus:
    """
        Classe représentant les données pertinentes par rapport à la tâche de dependency parsing obtenues à partir de la lecture d'un fichier contenant des arbres syntaxiques.

        Attributes:
            labels: L'ensemble des relations de dépendance entre une tête et son dépendant.
            tree_bank_file: Fichier au format conllu contenant les arbres syntaxiques de phrases.
            train_file: Fichier contenant les id des phrases pour le train set.
            test_file: Fichier contenant les id des phrases pour le test set.
            dev_file: Fichier contenant les id des phrases pour le dev set.
            word2i: Dictionnaire mot du vocabulaire -> indice.
            i2word: Liste des mots du vocabulaire de toutes les phrases du fichier des arbres.
            cat2i: Dictionnaire ayant pour clés les catégories grammaticales des mots du vocabulaire et comme valeurs leur indice.
            i2cat: Liste des catégories grammaticales des mots du vocabulaire.
            split: Dictionnaire contenant les ids de phrases pour chaque set (train, test, dev) avec pour clé le nom du set.
            data_set: Dictionnaire id de phrase -> liste de mots (instances de la classe Word) associés.
            data_set_split: Dictionnaire nom de set -> id de phrase -> liste des mots (instances de la classe Word) associés.


        Methods:
            read_file: Lis le fichier contenant les arbres syntaxiques, créé les structures d'indexation et les structures pertinentes en rapport aux phrases du corpus.
            create_arc_mode: Créé une instance de la classe ArcMode à partir des données du corpus.
        """
    def __init__(self, tree_bank_file, train_file, test_file, dev_file, labels=set()):
        self.labels = labels
        self.tree_bank_file = tree_bank_file
        self.train_file = train_file
        self.test_file = test_file
        self.dev_file = dev_file
        self.word2i = {}
        self.i2word = []
        self.cat2i = {}
        self.i2cat = []
        self.gold_trees = {}

    def read_file(self):
        # lecture des fichiers contenant les id de phrases pour chaque set du split qui sont stockés dans des listes
        f_test = open(self.test_file, 'r', encoding="utf-8")
        f_train = open(self.train_file, 'r', encoding="utf-8")
        f_dev = open(self.dev_file, 'r', encoding="utf-8")
        train, test, dev = [], [], []
        for line in f_test.readlines():
            test.append(line[:-1])
        for line in f_train.readlines():
            train.append(line[:-1])
        for line in f_dev.readlines():
            dev.append(line[:-1])

        # lecture du fichier contenant les arbres syntaxiques
        f = open(self.tree_bank_file, 'r', encoding="utf-8")

        split = defaultdict(lambda: [])
        sequoia = {}
        sequoia_split = defaultdict(lambda: {})
        for line in f.readlines()[1:]:
            if "sent_id" in line:
                sent_id = line.split()[3]
                phrase = []
                list_of_arcs = []
            elif "text =" in line:
                pass
            elif len(line.split()) > 0:
                # on récupère les informations importantes relatives à chaque mot
                sline = line.split()
                lemma = sline[2]
                cat = sline[4]
                form = sline[1]
                id = int(sline[0])
                head = int(sline[6])
                label = sline[7]
                self.labels.add(label)

                # création des arcs et des mots
                list_of_arcs.append(Arc(id, head, sline[7]))
                word = Word(id, form, lemma, cat, head, label)
                phrase.append(word)
                if lemma not in self.word2i:
                    self.word2i[lemma] = len(self.i2word)
                    self.i2word.append(lemma)
                if cat not in self.cat2i:
                    self.cat2i[cat] = len(self.i2cat)
                    self.i2cat.append(cat)
            else:
                # on ne s'occupe que des phrases dont l'arbre de dépendance est projectif
                if Tree(sent_id, list_of_arcs).is_projective():
                    sequoia[sent_id] = phrase
                    self.gold_trees[sent_id] = Tree(sent_id, list_of_arcs)
                    # on remplit le dictionnaire de split avec les id de phrases qui concernent chaque set
                    if sent_id in test:
                        sequoia_split['test'][sent_id] = phrase
                        split['test'].append(sent_id)
                    if sent_id in dev:
                        sequoia_split['dev'][sent_id] = phrase
                        split['dev'].append(sent_id)
                    if sent_id in train:
                        split['train'].append(sent_id)
                        sequoia_split['train'][sent_id] = phrase

        # ajouts des mots spéciaux pour les features du perceptron
        self.word2i["buffer_too_short"] = len(self.i2word)
        self.i2word.append("buffer_too_short")
        self.word2i["stack_too_short"] = len(self.i2word)
        self.i2word.append("stack_too_short")
        self.word2i["root"] = len(self.i2word)
        self.i2word.append("root")
        self.cat2i["buffer_too_short"] = len(self.i2cat)
        self.i2cat.append("buffer_too_short")
        self.cat2i["stack_too_short"] = len(self.i2cat)
        self.i2cat.append("stack_too_short")
        self.cat2i["root"] = len(self.i2cat)
        self.i2cat.append("root")

        # stocke les variables split, sequoia et sequoia split dans les attirbuts du corpus
        self.split = split
        self.data_set = sequoia
        self.data_set_split = sequoia_split

    def create_arc_mode(self, use_labels, arc_mode):
        if arc_mode == "standard":
            self.arc_mode = ArcStandard(use_labels, self.labels)
            return self.arc_mode
        elif arc_mode == "eager":
            self.arc_mode = ArcEager(use_labels, self.labels)
            return self.arc_mode
        else:
            return None


class NeuralNetMLP(nn.Module):

    def __init__(self, word_emb_size, cat_emb_size, hidden_layer_size, vocab_size, context_size, num_cat, num_labels,
                 corpus, with_pretrained=False):
        super(NeuralNetMLP, self).__init__()
        # lecture du fichier sérializé qui contient un dictionnaire mots (lemme ou forme) -> vecteur dense pré-entrainé
        with open('vecteurs_sequoia.pkl', 'rb') as fp:
            vecteurs_ = defaultdict(lambda: np.random.rand(300), pickle.load(fp))
        # initialisation d'un vecteur sparse pour représenter la catégorie, lequel sera concaténé à l'embedding de son lemme ou sa forme
        weight = np.array([vecteurs_[mot] for mot in corpus.i2word])
        weight_cat = np.array([[int(cat1 == cat2) for cat1 in corpus.i2cat] for cat2 in corpus.i2cat])

        if with_pretrained:
            self.word_emb_size = 300
            self.word_embs = nn.Embedding.from_pretrained(torch.FloatTensor(weight), freeze=False)
            self.cat_emb_size = len(corpus.i2cat)
            self.cat_embs = nn.Embedding.from_pretrained(torch.FloatTensor(weight_cat), freeze=False)
        else:
            self.word_emb_size = word_emb_size
            self.word_embs = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_emb_size)
            self.cat_emb_size = cat_emb_size
            self.cat_embs = nn.Embedding(num_embeddings=num_cat, embedding_dim=cat_emb_size)

        self.context_size = context_size
        self.hidden_layer_size = hidden_layer_size
        self.linear1 = nn.Linear(2 * self.context_size * (self.word_emb_size + self.cat_emb_size), self.hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, num_labels)

    def forward(self, X):
        """
            Input : Concaténation des word embeddings des mots considérés du buffer et de la stack ainsi
            que des embeddings de leur catégorie.

            Output : Log-probabilités des actions pour chaque exemple.

        """
        X = [x.view(1, -1) for x in X]
        X = torch.cat(X, dim=0)
        input = torch.cat((self.word_embs(X[:, :2*self.context_size]), self.cat_embs(X[:, 2*self.context_size:])), dim=2)
        input = input.view(-1, 2*self.context_size * (self.word_emb_size + self.cat_emb_size))
        out = self.linear1(input)
        out = torch.relu(out)
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)


class Classifier:
    """
        Classe abstraite qui représente l'architecture des classifieurs.

    """

    def feature_extraction(self, config):
        pass

    def train(self, nb_epochs):
        pass

    def get_classifier(self):
        pass

    def set_classifier(self, clf):
        pass

    def evaluate(self, set):
        pass

    def predicted_tree(self, sentid):
        pass

    def attachment_score(self, set):
        pass

    def tree_score_and_errors(self, set, n):
        pass

    def error_analysis_transition(self, set, n):
        pass

    def pretty_print_tree(self, sent_id):
        tree = self.predicted_tree(sent_id)
        tree = sorted(tree, key=lambda word: word.id)
        for word in tree:
            print(word)


class P(Classifier):
    """
            Classe représentant un Perceptron utilisé dans le cadre du dependency parsing.

            Attributes:
                perceptron: Instance de la classe Perceptron de sklearn.
                arc_mode: Nom du type d'arc utilisé ("standard" our "eager").
                corpus: Instance de la classe corpus.
                vec: Instance de la classe DictVectorizer utilisé pour les features du Perceptron.

            Methods:
                feature_extraction: Extraction des features pour les fournir au Perceptron.
                train: Entrainement du Perceptron.
                evaluate: Evalue la macro accuracy et la micro accuracy obtenues avec les prédictions du Perceptron sur un ensemble d'exemples.
                predicted_tree: Predit l'arbre de dépendance d'une phrase avec le Percpetron comme classifieur pour prédire la transition pour chaque configuration.
                attachment_score: Calcule les métriques UAS et LAS.

            """
    def __init__(self, corpus, arc_mode):
        self.perceptron = Perceptron(max_iter=1, warm_start=True)
        self.arc_mode = arc_mode
        self.corpus = corpus
        self.vec = DictVectorizer()

    def get_classifier(self):
        return self.perceptron

    def set_classifier(self, clf):
        self.perceptron = clf

    def feature_extraction(self, config):
        features = {}
        if len(config.stack) >= 1:
            features['s1_form'] = config.stack[-1].form
            features['s1_lemma'] = config.stack[-1].lemma
            features['s1_xpos'] = config.stack[-1].xpos
        if len(config.stack) >= 2:
            features['s2_form'] = config.stack[-2].form
            features['s2_lemma'] = config.stack[-2].lemma
            features['s2_xpos'] = config.stack[-2].xpos
        if len(config.buffer) >= 1:
            features['b1_form'] = config.buffer[0].form
            features['b1_lemma'] = config.buffer[0].lemma
            features['b1_xpos'] = config.buffer[0].xpos
        if len(config.buffer) >= 2:
            features['b2_form'] = config.buffer[1].form
            features['b2_lemma'] = config.buffer[1].lemma
            features['b2_xpos'] = config.buffer[1].xpos
        features['biais'] = 'biais'
        return features

    def train(self, nb_epochs):
        corpus = self.corpus

        NB_EPOCHS = nb_epochs
        train_accuracies = []
        dev_accuracies = []

        # Initialize the perceptron
        perceptron = self.perceptron

        # Get features from training and development data
        X_train, y_train = corpus.arc_mode.training_oracle(corpus, "train")
        X_dev, y_dev = corpus.arc_mode.training_oracle(corpus, "dev")
        X_train = [self.feature_extraction(x) for x in X_train]
        X_dev = [self.feature_extraction(x) for x in X_dev]
        y_train = [y.transition for y in y_train]
        y_dev = [y.transition for y in y_dev]


        X_train = self.vec.fit_transform(X_train)
        X_dev = self.vec.transform(X_dev)

        for epoch in range(NB_EPOCHS):
            # Train the perceptron for one epoch
            perceptron.fit(X_train, y_train)

            # Evaluate perceptron performance on the training set
            y_pred_train = perceptron.predict(X_train)
            train_accuracies.append(f1_score(y_train, y_pred_train, average="macro"))

            # Evaluate perceptron performance on the validation set
            y_pred_dev = perceptron.predict(X_dev)
            dev_accuracies.append(f1_score(y_dev, y_pred_dev, average="macro"))

            # Check early stopping condition
            if epoch > 1:
                if (dev_accuracies[-1] < dev_accuracies[-2] < dev_accuracies[-3]
                        and train_accuracies[-3] < train_accuracies[-2] < train_accuracies[-1]):
                    print(f"EARLY STOPPING: EPOCH = {epoch}")
                    break

        abs = [i for i in range(len(train_accuracies))]
        plt.plot(abs, train_accuracies, label='train accuracy')
        plt.plot(abs, dev_accuracies, label='dev accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

    def evaluate(self, set):
        X, Y = self.corpus.arc_mode.training_oracle(self.corpus, set)
        X = [self.feature_extraction(x) for x in X]
        Y = [y.transition for y in Y]
        X = self.vec.transform(X)
        y_pred = self.perceptron.predict(X)
        print("macro : ", f1_score(Y, y_pred, average="macro"))
        print("micro : ", f1_score(Y, y_pred, average="micro"))

    def predicted_tree(self, sentid):
        stack = [Word(0, 'root', 'root', 'root', -1, '')]
        buffer = [word.headless() for word in self.corpus.data_set[sentid]]
        previous_arcs = []
        configuration = Configuration(stack, buffer, previous_arcs)
        while ((configuration.buffer or configuration.stack) and self.arc_mode.type == 'standard') or (configuration.buffer and self.arc_mode.type == 'eager'):
            features = self.feature_extraction(configuration)
            features = self.vec.transform(features)
            predictions = self.perceptron.decision_function(features)[0]
            classes = self.perceptron.classes_
            configuration = Configuration(configuration.stack, configuration.buffer, configuration.previous_arcs)
            for indice in np.argsort(-predictions):
                #print(indice)
                transition = self.corpus.arc_mode.transition_name2transition[classes[indice]]
                if transition.check(configuration):
                    transition.apply(configuration)
                    break
        return configuration.previous_arcs

    def attachment_score(self, set):
        well_predicted_labelled_arcs = []
        well_predicted_unlabelled_arcs = []
        total_arcs = 0
        for sentid in self.corpus.split[set]:
            predictions = self.predicted_tree(sentid)
            sentence = self.corpus.data_set[sentid]
            total_arcs += len(sentence)
            for word in sentence:
                for prediction in predictions:
                    if word.id == prediction.id:
                        if word.head == prediction.head:
                            well_predicted_unlabelled_arcs.append(word.id)
                            if word.deprel == prediction.deprel:
                                well_predicted_labelled_arcs.append(word.id)
                        continue
        print(f"unlabeled attachment score on {set} : {len(well_predicted_unlabelled_arcs) / total_arcs}")
        print(f"labeled attachment score on {set} : {len(well_predicted_labelled_arcs) / total_arcs}")


class MLP(Classifier):
    """
        Classe représentant un MLP utilisé dans le cadre du dependency parsing.

        Attributes:
            mlp: Instance de la classe NeuralNetMLP (qui hérite de nn.Module).
            arc_mode: Nom du type d'arc utilisé ("standard" our "eager").
            corpus: Instance de la classe Corpus.
            context_size: Nombre de mots de la stack et du buffer que l'on prend en compte pour les features pour le MLP (on prend donc en compte 2*context_size mots).
            batch_size: Taille des batch.
            lr: Learning rate du MLP.

        Methods:
            feature_extraction: Extraction des features pour les fournir au MLP.
            train: Entrainement du MLP.
            evaluate: Evalue la macro accuracy et la micro accuracy obtenues avec les prédictions du MLP sur un ensemble d'exemples.
            predicted_tree: Predit l'arbre de dépendance d'une phrase avec le MLP comme classifieur pour prédire la transition pour chaque configuration.
            tree_score_and_errors: Calcule les métriques UAS et LAS, calcule les erreurs de prédiction les plus courantes sur les relations de dépendance.
            error_analysis_transition: Calcule les erreurs de prédiction les plus courantes sur les transitions.

    """
    def __init__(self, corpus, arc_mode, BATCH_SIZE = 25, patience=2, lr=0.0005, WORD_EMB_SIZE=300, CAT_EMB_SIZE=20, HIDDEN_LAYER_SIZE=150, CONTEXT_SIZE=3):
        self.mlp = NeuralNetMLP(word_emb_size=WORD_EMB_SIZE,
                                cat_emb_size=CAT_EMB_SIZE,
                                hidden_layer_size=HIDDEN_LAYER_SIZE,
                                vocab_size=len(corpus.word2i),
                                context_size=CONTEXT_SIZE,
                                num_cat=len(corpus.cat2i),
                                num_labels=len(arc_mode.transition2i),
                                with_pretrained=True,
                                corpus=corpus)
        self.context_size = CONTEXT_SIZE
        self.corpus = corpus
        self.arc_mode = arc_mode
        self.batch_size = BATCH_SIZE
        self.lr = lr
        self.patience = patience

    def get_classifier(self):
        return self.mlp

    def set_classifier(self, clf):
        self.mlp = clf

    def feature_extraction_neural_net(self, config):
        word2i = self.corpus.word2i
        cat2i = self.corpus.cat2i
        words = [word2i[config.stack[-k].lemma] if len(config.stack) >= k else word2i["stack_too_short"] for k in range(1, self.context_size + 1)] + \
                [word2i[config.buffer[k].lemma] if len(config.buffer) > k else word2i["buffer_too_short"] for k in range(self.context_size)]
        cats = [cat2i[config.stack[-k].xpos] if len(config.stack) >= k else cat2i["stack_too_short"] for k in range(1, self.context_size + 1)] + \
               [cat2i[config.buffer[k].xpos] if len(config.buffer) > k else cat2i["buffer_too_short"] for k in range(self.context_size)]
        return words + cats

    def convert_examples_to_tensors(self, X, Y):
        fX = torch.LongTensor([self.feature_extraction_neural_net(config) for config in X])
        fY = torch.LongTensor([self.arc_mode.transition2i[gold_action.transition] for gold_action in Y])
        return fX, fY

    def convert_example_to_tensors(self, config, transition):
        fX = torch.LongTensor(self.feature_extraction_neural_net(config))
        fY = torch.LongTensor(self.arc_mode.transition2i[transition.transition])
        return fX, fY

    def train(self, nb_epochs):
        NB_EPOCHS = nb_epochs
        BATCH_SIZE = self.batch_size
        train_losses = []
        dev_losses = []
        dev_uascores = []
        dev_lascores = []

        loss_function = nn.NLLLoss()

        action_predictor = self.mlp
        optimizer = optim.Adam(action_predictor.parameters(), lr=self.lr)

        X, y, fX, fy = {}, {}, {}, {}
        for set in ['train', 'dev']:
            X[set], y[set] = self.corpus.arc_mode.training_oracle(self.corpus, set)
            fX[set], fy[set] = self.convert_examples_to_tensors(X[set], y[set])

        for epoch in range(NB_EPOCHS):
            print(epoch)
            epoch_loss = 0

            examples = list(zip(fX['train'], fy['train']))
            shuffle(examples)
            non_tensor_fX_train, non_tensor_fY_train = zip(*examples)

            i = 0
            while i < len(non_tensor_fX_train):
                batch_X = non_tensor_fX_train[i: i + BATCH_SIZE]
                batch_Y = torch.LongTensor(non_tensor_fY_train[i: i + BATCH_SIZE])
                i += BATCH_SIZE

                action_predictor.zero_grad()
                log_probs = action_predictor(batch_X)
                loss = loss_function(log_probs, batch_Y)
                loss.backward()
                optimizer.step()
                # epoch_loss += loss.item()
            # train_losses.append(epoch_loss)

            log_probs_train = action_predictor(fX['train'])
            loss_train = loss_function(log_probs_train, fy['train']).detach().numpy()
            train_losses.append(loss_train)

            log_probs_dev = action_predictor(fX['dev'])
            loss_dev = loss_function(log_probs_dev, fy['dev']).detach().numpy()
            dev_losses.append(loss_dev)

            dev_uas, dev_las = self.tree_score_and_errors("dev", 0)
            dev_uascores.append(dev_uas)
            dev_lascores.append(dev_las)
            """
            if epoch >= 2:
                if dev_losses[-1] > dev_losses[-2] > dev_losses[-3] and train_losses[-3] > train_losses[-2] > train_losses[-1]:
                    print(f"EARLY STOPPING: EPOCH = {epoch}")
                    break
            """
            # early stopping
            if self.arc_mode.use_labels:
                if epoch >= self.patience:
                    if all(dev_lascores[i] > dev_lascores[i - 1] for i in range(-1, -self.patience-1, -1)):
                        print(f"EARLY STOPPING: EPOCH = {epoch}")
                        break
            else:
                if epoch >= self.patience:
                    if all(dev_uascores[i] > dev_uascores[i - 1] for i in range(-1, -self.patience-1, -1)):
                        print(f"EARLY STOPPING: EPOCH = {epoch}")
                        break

        abs = [i for i in range(len(train_losses))]
        plt.plot(abs, train_losses, label='train loss')
        plt.plot(abs, dev_losses, label='dev loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def dynamic_train(self, nb_epochs):
        NB_EPOCHS = nb_epochs
        # BATCH_SIZE = self.batch_size
        train_losses = []
        dev_losses = []
        dev_uascores = []
        dev_lascores = []

        loss_function = nn.NLLLoss()

        action_predictor = self.mlp
        optimizer = optim.Adam(action_predictor.parameters(), lr=self.lr)

        if self.corpus.arc_mode.type == "eager":
            train_examples = self.corpus.arc_mode.dynamic_training_oracle(self.corpus, "train")
            dev_examples = self.corpus.arc_mode.dynamic_training_oracle(self.corpus, "dev")
        else:
            print("What are you doing ?!")
            return None

        X_train, y_train, fX_train, fy_train = {}, {}, {}, {}
        for sent_id in self.corpus.data_set_split["train"]:
            X_train[sent_id] = [example.config for example in train_examples[sent_id]]
            y_train[sent_id] = [example.transition for example in train_examples[sent_id]]
            fX_train[sent_id], fy_train[sent_id] = self.convert_examples_to_tensors(X_train[sent_id], y_train[sent_id])

        X_dev, y_dev, fX_dev, fy_dev = {}, {}, {}, {}
        for sent_id in self.corpus.data_set_split["dev"]:
            X_dev[sent_id] = [example.config for example in dev_examples[sent_id]]
            y_dev[sent_id] = [example.transition for example in dev_examples[sent_id]]
            fX_dev[sent_id], fy_dev[sent_id] = self.convert_examples_to_tensors(X_dev[sent_id], y_dev[sent_id])

        for epoch in range(NB_EPOCHS):
            print(epoch)
            epoch_loss = 0
            for sent_id in X_train:
                previous_arcs = []
                stack = [Word(0, 'root', 'root', 'root', -1, '')]
                buffer = [word for word in self.corpus.data_set[sent_id]]
                config = Configuration(stack, buffer, previous_arcs)
                pred_transitions = []
                gold_transitions = []
                while buffer:
                    pred_scores = action_predictor(config)
                    for indice in torch.argsort(-pred_scores)[0]:
                        # print(indice)
                        transition = self.corpus.arc_mode.transition_name2transition[self.corpus.arc_mode.i2transition[indice]]
                        if transition.check(config):
                            pred_transitions.append(transition)
                            break
                    if transition.cost(config, self.corpus.gold_trees[sent_id].list_of_arcs) == 0:
                        gold_transitions.append(transition)
                        transition.apply(config)
                    else:
                        b = config.buffer[0]
                        s = config.stack[-1]
                        zero_cost_transitions = []
                        right_arc = self.corpus.arc_mode.transition_name2transition[f"right_arc{self.corpus.arc_mode.use_labels * ('_' + b.deprel)}"]
                        left_arc = self.corpus.arc_mode.transition_name2transition[f"left_arc{self.corpus.arc_mode.use_labels * ('_' + s.deprel)}"]
                        reduce = self.corpus.arc_mode.transition_name2transition["reduce"]
                        shift = self.corpus.arc_mode.transition_name2transition["shift"]
                        done = self.corpus.arc_mode.transition_name2transition["done"]
                        gold_arcs = self.corpus.gold_trees[sent_id].list_of_arcs
                        if right_arc.check(config):
                            if right_arc.cost(config, gold_arcs) == 0:
                                zero_cost_transitions.append(right_arc)
                        if left_arc.check(config):
                            if left_arc.cost(config, gold_arcs) == 0:
                                zero_cost_transitions.append(left_arc)
                        if shift.check(config):
                            if shift.cost(config, gold_arcs) == 0:
                                zero_cost_transitions.append(shift)
                        if reduce.check(config):
                            if reduce.cost(config, gold_arcs) == 0:
                                zero_cost_transitions.append(reduce)
                        if done.check(config):
                            zero_cost_transitions.append(done)
                        random.choice(zero_cost_transitions).apply(config)


            examples = list(zip(fX['train'], fy['train']))
            shuffle(examples)
            non_tensor_fX_train, non_tensor_fY_train = zip(*examples)

            i = 0
            while i < len(non_tensor_fX_train):
                batch_X = non_tensor_fX_train[i: i + BATCH_SIZE]
                batch_Y = torch.LongTensor(non_tensor_fY_train[i: i + BATCH_SIZE])
                i += BATCH_SIZE

                action_predictor.zero_grad()
                log_probs = action_predictor(batch_X)
                loss = loss_function(log_probs, batch_Y)
                loss.backward()
                optimizer.step()
                # epoch_loss += loss.item()
            # train_losses.append(epoch_loss)

            log_probs_train = action_predictor(fX['train'])
            loss_train = loss_function(log_probs_train, fy['train']).detach().numpy()
            train_losses.append(loss_train)

            log_probs_dev = action_predictor(fX['dev'])
            loss_dev = loss_function(log_probs_dev, fy['dev']).detach().numpy()
            dev_losses.append(loss_dev)

            dev_uas, dev_las = self.tree_score_and_errors("dev", 0)
            dev_uascores.append(dev_uas)
            dev_lascores.append(dev_las)
            """
            if epoch >= 2:
                if dev_losses[-1] > dev_losses[-2] > dev_losses[-3] and train_losses[-3] > train_losses[-2] > train_losses[-1]:
                    print(f"EARLY STOPPING: EPOCH = {epoch}")
                    break
            """
            # early stopping
            if self.arc_mode.use_labels:
                if epoch >= self.patience:
                    if all(dev_lascores[i] > dev_lascores[i - 1] for i in range(-1, -self.patience - 1, -1)):
                        print(f"EARLY STOPPING: EPOCH = {epoch}")
                        break
            else:
                if epoch >= self.patience:
                    if all(dev_uascores[i] > dev_uascores[i - 1] for i in range(-1, -self.patience - 1, -1)):
                        print(f"EARLY STOPPING: EPOCH = {epoch}")
                        break

        abs = [i for i in range(len(train_losses))]
        plt.plot(abs, train_losses, label='train loss')
        plt.plot(abs, dev_losses, label='dev loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def evaluate(self, set):
        X, y, fX, fy, y_pred = {}, {}, {}, {}, {}
        X[set], y[set] = self.corpus.arc_mode.training_oracle(self.corpus, set)
        fX[set], fy[set] = self.convert_examples_to_tensors(X[set], y[set])
        y_pred[set] = torch.argmax(self.mlp(fX[set]), dim=1)
        log_probs = self.mlp(fX[set])
        loss_function=nn.NLLLoss()
        loss = loss_function(log_probs, fy[set]).detach().numpy()
        print(f"loss: {loss}")
        # conf_mat = ConfusionMatrix(task='multiclass', num_classes=len(self.corpus.arc_mode.i2transition))
        # print(f"matrice de confusion pour le {set} : \n {conf_mat(fy[set], y_pred[set])} \n\n")
        print(f"Sur le {set} : \n micro : {f1_score(fy[set], y_pred[set], average = 'micro')} \n macro : {f1_score(fy[set], y_pred[set], average = 'macro')} \n")

    def predicted_tree(self, sentid):
        stack = [Word(0, 'root', 'root', 'root', -1, '')]
        buffer = [word.headless() for word in self.corpus.data_set[sentid]]
        previous_arcs = []
        configuration = Configuration(stack, buffer, previous_arcs)
        while ((configuration.buffer or configuration.stack) and self.arc_mode.type == 'standard') or (configuration.buffer and self.arc_mode.type == 'eager') :
            features = torch.LongTensor([self.feature_extraction_neural_net(configuration)])
            predictions = self.mlp(features)
            classes = self.corpus.arc_mode.i2transition
            configuration = Configuration(configuration.stack, configuration.buffer, configuration.previous_arcs)
            for indice in torch.argsort(-predictions)[0]:
                #print(indice)
                transition = self.corpus.arc_mode.transition_name2transition[classes[indice]]
                if transition.check(configuration):
                    transition.apply(configuration)
                    break
        return configuration.previous_arcs

    def tree_score_and_errors(self, set, n):
        well_predicted_labelled_arcs = []
        well_predicted_unlabelled_arcs = []
        errors_labels = []
        total_arcs = 0
        for sentid in self.corpus.split[set]:
            predictions = self.predicted_tree(sentid)
            sentence = self.corpus.data_set[sentid]
            total_arcs += len(sentence)
            for word in sentence:
                for prediction in predictions:
                    if word.id == prediction.id:
                        if word.head == prediction.head:
                            well_predicted_unlabelled_arcs.append(word.id)
                            if word.deprel == prediction.deprel:
                                well_predicted_labelled_arcs.append(word.id)
                            else :
                                errors_labels.append((word.deprel, prediction.deprel))
                        continue

        print(f"unlabeled attachment score on {set} : {len(well_predicted_unlabelled_arcs) / total_arcs}")
        print(f"labeled attachment score on {set} : {len(well_predicted_labelled_arcs) / total_arcs}")
        print(f"{n} erreurs les plus courrantes sur les labels : {Counter(errors_labels).most_common(n)}")
        uas = len(well_predicted_unlabelled_arcs) / total_arcs
        las = len(well_predicted_labelled_arcs) / total_arcs
        return uas, las

    def error_analysis_transition(self, set, n):
        errors = []
        X, y, fX, fy, y_pred = {}, {}, {}, {}, {}
        X[set], y[set] = self.corpus.arc_mode.training_oracle(self.corpus, set)
        fX[set], fy[set] = self.convert_examples_to_tensors(X[set], y[set])
        y_pred[set] = torch.argmax(self.mlp(fX[set]), dim=1)
        for gold, predicted in zip(fy[set].tolist(), y_pred[set].tolist()):
            if gold != predicted:
                errors.append((self.arc_mode.i2transition[gold], self.arc_mode.i2transition[predicted]))
        print(f"{n} erreurs les plus courrantes sur les transitions : {Counter(errors).most_common(n)}")


class Configuration:
    """
        Classe représentant une configuration.

        Attributes:
            stack: Liste de mots (instances de Word) représentant l'état de la stack.
            buffer: Liste de mots (instances de Word) représentant l'état du buffer.
            previous_arcs: Liste de mots (instances de Word) ayant déjà reçu une tête (et une relation de dépendance).

    """

    def __init__(self, stack, buffer, previous_arcs):
        self.stack = [mot for mot in stack]
        self.buffer = [mot for mot in buffer]
        self.previous_arcs = [mot for mot in previous_arcs]

    def copy(self):
        new_stack = [word.copy() for word in self.stack]
        new_buffer = [word.copy() for word in self.buffer]
        new_previous_arcs = [word.copy() for word in self.previous_arcs]
        return Configuration(new_stack, new_buffer, new_previous_arcs)


class Transition:
    """
        Classe représentant une transition.

        Attributes:
            arc_type: Nom du mode d'arc utilisé ("standard" our "eager").
            transition: Etiquette de la transition.

        Methods:
            check: Vérifie les conditions d'application de la transition sur une configuration donnée.
            apply: Applique la transition sur une configuration donnée si cela est possible.

    """

    def __init__(self, arc_type, transition):
        self.arc_type = arc_type
        self.transition = transition

    def check(self, config):
        pass

    def apply(self, config):
        pass

    def cost(self,configuration, gold_arcs):
        pass

class RightArc(Transition):

    def __init__(self, arc_type, label):
        use_labels = True if label else False
        super(RightArc, self).__init__(arc_type, f"right_arc{use_labels*'_'+label}")
        self.label = label

    def check(self, config):
        if self.arc_type == "standard":
            return len(config.stack) >= 2 and config.stack[-1].head is None
        else:
            return len(config.stack) >= 1 and len(config.buffer) >= 1 and config.buffer[0].head is None

    def apply(self, config):
        if self.arc_type == "standard":
            dependant = config.stack.pop(-1)
            head = config.stack[-1]
            dependant.head = head.id
            dependant.deprel = self.label
            config.previous_arcs.append(dependant)
        elif self.arc_type == "eager":
            dependant = config.buffer[0]
            head = config.stack[-1]
            dependant.head = head.id
            dependant.deprel = self.label
            config.stack.append(config.buffer.pop(0))
            config.previous_arcs.append(dependant)

    def cost(self,configuration, gold_arcs):
        b = configuration.buffer[0]
        C = 0
        for k in configuration.buffer[1:] + configuration.stack[:-1]:
            for arc in gold_arcs:
                if (arc.head == k and arc.dependent == b):
                    C += 1
        for k in configuration.stack[:-1]:
            for arc in gold_arcs:
                if (arc.head == b and arc.dependent == k):
                    C += 1
        return C

class LeftArc(Transition):

    def __init__(self, arc_type, label):
        use_labels = True if label else False
        self.label = label
        super(LeftArc, self).__init__(arc_type, f"left_arc{use_labels*'_'+label}")

    def check(self, config):
        if self.arc_type == "standard":
            return len(config.stack) >= 2 and config.stack[-2].head is None
        elif self.arc_type == "eager":
            return len(config.stack) >= 1 and len(config.buffer) >= 1 and config.stack[-1].head is None

    def apply(self, config):
        if self.arc_type == "standard":
            dependant = config.stack.pop(-2)
            head = config.stack[-1]
            dependant.head = head.id
            dependant.deprel = self.label
            config.previous_arcs.append(dependant)
        elif self.arc_type == "eager":
            dependant = config.stack.pop(-1)
            head = config.buffer[0]
            dependant.head = head.id
            dependant.deprel = self.label
            config.previous_arcs.append(dependant)

    def cost(self,configuration, gold_arcs):
        s = configuration.stack[-1]
        C = 0
        for k in configuration.buffer[1:]:
            for arc in gold_arcs:
                if (arc.head == k and arc.dependent == s) or (arc.head == s and arc.dependent == k):
                    C += 1
        return C



class Shift(Transition):
    def __init__(self, arc_type):
        super(Shift, self).__init__(arc_type, "shift")

    def check(self, config):
        return len(config.buffer) != 0


    def apply(self, config):
        config.stack.append(config.buffer.pop(0))

    def cost(self,configuration, gold_arcs):
        b = configuration.buffer[0]
        C = 0
        for k in configuration.stack:
            for arc in gold_arcs:
                if (arc.head == k and arc.dependent == b) or (arc.head == b and arc.dependent == k):
                    C += 1
        return C


class Reduce(Transition):
    def __init__(self, arc_type):
        super(Reduce, self).__init__(arc_type, "reduce")

    def check(self, config):
        return len(config.stack) >= 1 and config.stack[-1].head is not None

    def apply(self, config):
        config.stack.pop(-1)

    def cost(self,configuration, gold_arcs):
        s = configuration.stack[-1]
        C = 0
        for k in configuration.buffer:
            for arc in gold_arcs:
                if (arc.head == s and arc.dependent == k):
                    C += 1
        return C


class Done(Transition):
    def __init__(self, arc_type):
        super(Done, self).__init__(arc_type, "done")

    def check(self, config):
        if self.arc_type == "standard":
            return len(config.stack) == 1 and len(config.buffer) == 0
        else:
            pass

    def apply(self, config):
        if self.arc_type == "standard":
            config.stack.pop(-1)
        else:
            pass


class Example:

    def __init__(self, config, transition):
        self.config = config
        self.transition = transition


class Word:
    """
        Classe représentant un mot et contenant toutes les informations pertinentes du corpus pour celui-ci.

        Attributes:
            id: id du mot dans la phrase.
            form: Forme du mot.
            lemma: Lemme associé au mot.
            xpos: Catégorie grammaticale.
            head: Tête du mot.
            deprel: Relation de dépendance.

        Methods:
            copy: Crée une copie du mot.
            headless: Crée une copie du mot sans tête ni relation de dépendance.

    """

    def __init__(self, id, form, lemma, xpos, head, deprel):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.xpos = xpos
        self.head = head
        self.deprel = deprel

    def copy(self):
        return Word(self.id, self.form, self.lemma, self.xpos, self.head, self.deprel)

    def __str__(self):
        return f"{self.id} {self.form} {self.head} {self.deprel}"

    def headless(self):
        return Word(self.id, self.form, self.lemma, self.xpos, None, None)


class ArcMode:

    def __init__(self, labels):
        self.labels = labels

    def get_transitions_set(self):
        pass

    def training_oracle_sentence(self, id_sent, corpus):
        pass

    def training_oracle(self, corpus, set):
        pass


class ArcStandard(ArcMode):
    """
        Classe représentant un mode d'arc standard pour le training oracle.

        Attributes:
            use_labels: Utilisation des relations de dépendance ou non dans les arcs.
            labels: Relations de dépendance présentes dans un corpus.
            type: Nom du mode d'arc.
            transitions: Ensemble des étiquettes des transitions associées au mode d'arc.
            transition_name2transition: Dictionnaire étiquette de transition -> instance de la classe Transition associée.
            i2transition: Liste des étiquettes des transitions associées au mode d'arc.
            transition2i: Dictionnaire étiquette de transition -> indice.

        Methods:
            training_oracle_sentence: Applique le training oracle arc standard sur une phrase pour obtenir la liste d'exmples à partir de cette phrase.
            training_oracle: Applique le training oracle arc standard sur toutes les phrases d'un set (train, test ou dev).

    """

    def __init__(self, use_labels=False, labels=set()):
        super(ArcStandard, self).__init__(labels)
        self.use_labels = use_labels
        self.type = 'standard'
        self.transition_name2transition = {}
        if use_labels:
            self.transitions = {'shift', 'done'}
            self.transition_name2transition['shift'] = Shift(self.type)
            self.transition_name2transition['done'] = Done(self.type)
            for label in self.labels:
                self.transitions.add('right_arc_' + label)
                self.transition_name2transition['right_arc_' + label] = RightArc(self.type, label)
                self.transitions.add('left_arc_' + label)
                self.transition_name2transition['left_arc_' + label] = LeftArc(self.type, label)

        else:
            self.transitions = {'shift', 'done', 'left_arc', 'right_arc'}
            self.transition_name2transition['shift'] = Shift(self.type)
            self.transition_name2transition['done'] = Done(self.type)
            self.transition_name2transition['right_arc'] = RightArc(self.type, '')
            self.transition_name2transition['left_arc'] = LeftArc(self.type, '')
        self.i2transition = list(self.transitions)
        self.transition2i = {transition: i for i, transition in enumerate(self.i2transition)}

    def get_transitions_set(self):
        return self.transitions

    def training_oracle_sentence(self, sent_id, corpus):
        stack = [Word(0, 'root', 'root', 'root', -1, '')]
        buffer = [word for word in corpus.data_set[sent_id]]
        previous_arcs = []
        examples = [Example(Configuration(stack, buffer, previous_arcs), self.transition_name2transition["shift"])]
        stack.append(buffer[0]), buffer.pop(0)
        while buffer or stack:
            if 2 > len(stack):
                print(f"ERREUR_{sent_id}")
                break
            else:
                right = stack[-1]
                left = stack[-2]
                if left.head == right.id:
                    examples.append(Example(Configuration(stack, buffer, previous_arcs), self.transition_name2transition[f"left_arc{self.use_labels*('_'+left.deprel)}"]))
                    previous_arcs.append(stack.pop(-2))
                elif right.head == left.id and all(
                        [right.id != word.head for word in stack + buffer]):
                    examples.append(Example(Configuration(stack, buffer, previous_arcs), self.transition_name2transition[f"right_arc{self.use_labels*('_'+right.deprel)}"]))
                    previous_arcs.append(stack.pop(-1))
                    if len(stack) == 1 and len(buffer) == 0:  # Le done aura toujours lieu après un arc droit
                        examples.append(Example(Configuration(stack, buffer, previous_arcs), self.transition_name2transition["done"]))
                        previous_arcs.append(stack.pop(-1))
                elif len(buffer) != 0:
                    examples.append(Example(Configuration(stack, buffer, previous_arcs), self.transition_name2transition["shift"]))
                    stack.append(buffer[0])
                    buffer.pop(0)
                else:
                    print(f"ERREUR_{sent_id}")
                    return []
        return examples

    def training_oracle(self, corpus, set):
        list_of_examples = []
        for sentence_id in corpus.split[set]:
            sentence_examples = self.training_oracle_sentence(sentence_id, corpus)
            if sentence_examples:
                list_of_examples.append(sentence_examples)
        X, y = [], []
        for sentence_examples in list_of_examples:
            for example in sentence_examples:
                X.append(Configuration(example.config.stack, example.config.buffer, example.config.previous_arcs))
                y.append(example.transition)
        return X, y


class ArcEager(ArcMode):
    """
            Classe représentant un mode d'arc eager pour le training oracle.

            Attributes:
                use_labels: Utilisation des relations de dépendance ou non dans les arcs.
                labels: Relations de dépendance présentes dans un corpus.
                type: Nom du mode d'arc.
                transitions: Ensemble des étiquettes des transitions associées au mode d'arc.
                transition_name2transition: Dictionnaire étiquette de transition -> instance de la classe Transition associée.
                i2transition: Liste des étiquettes des transitions associées au mode d'arc.
                transition2i: Dictionnaire étiquette de transition -> indice.

            Methods:
            training_oracle_sentence: Applique le training oracle arc eager sur une phrase pour obtenir la liste d'exmples à partir de cette phrase.
            training_oracle: Applique le training oracle arc eager sur toutes les phrases d'un set (train, test ou dev).

        """

    def __init__(self, use_labels=False, labels=set()):
        super(ArcEager, self).__init__(labels)
        self.type = 'eager'
        self.use_labels = use_labels
        self.transition_name2transition = {}
        if use_labels:
            self.transitions = {'shift', 'reduce'}
            self.transition_name2transition['shift'] = Shift(self.type)
            self.transition_name2transition['reduce'] = Reduce(self.type)
            for label in self.labels:
                self.transitions.add('right_arc_' + label)
                self.transition_name2transition['right_arc_' + label] = RightArc(self.type, label)
                self.transitions.add('left_arc_' + label)
                self.transition_name2transition['left_arc_' + label] = LeftArc(self.type, label)
        else:
            self.transitions = {'shift', 'left_arc', 'right_arc', 'reduce'}
            self.transition_name2transition['shift'] = Shift(self.type)
            self.transition_name2transition['reduce'] = Reduce(self.type)
            self.transition_name2transition['right_arc'] = RightArc(self.type, '')
            self.transition_name2transition['left_arc'] = LeftArc(self.type, '')
        self.i2transition = list(self.transitions)
        self.transition2i = {transition : i for i, transition in enumerate(self.i2transition)}

    def get_transitions_set(self):
        return self.transitions

    def training_oracle_sentence(self, sent_id, corpus):
        stack = [Word(0, 'root', 'root', 'root', -1, '')]
        buffer = [word for word in corpus.data_set[sent_id]]
        previous_arcs = []
        examples = []
        while buffer:
            last_stack = stack[-1]
            first_buffer = buffer[0]
            if last_stack.head == first_buffer.id:
                examples.append(Example(Configuration(stack, buffer, previous_arcs), self.transition_name2transition[f"left_arc{self.use_labels*('_'+last_stack.deprel)}"]))
                previous_arcs.append(stack.pop(-1))
            elif first_buffer.head == last_stack.id:
                examples.append(Example(Configuration(stack, buffer, previous_arcs), self.transition_name2transition[f"right_arc{self.use_labels*('_'+first_buffer.deprel)}"]))
                stack.append(buffer.pop(0))
                previous_arcs.append(first_buffer)
            elif any([first_buffer.id == mot.head or mot.id == first_buffer.head for mot in stack]):
                examples.append(Example(Configuration(stack, buffer, previous_arcs), self.transition_name2transition["reduce"]))
                stack.pop(-1)
            elif len(buffer) != 0:
                examples.append(Example(Configuration(stack, buffer, previous_arcs), self.transition_name2transition["shift"]))
                stack.append(buffer.pop(0))
        return examples

    def training_oracle(self, corpus, set):
        list_of_examples = []
        for sentence_id in corpus.split[set]:
            sentence_examples = self.training_oracle_sentence(sentence_id, corpus)
            if sentence_examples:
                list_of_examples.append(sentence_examples)
        X, y = [], []
        for sentence_examples in list_of_examples:
            for example in sentence_examples:
                X.append(Configuration(example.config.stack, example.config.buffer, example.config.previous_arcs))
                y.append(example.transition)
        return X, y

    def training_dynamic_oracle(self, corpus, set):
        # configuration
        examples = {}
        for sent_id in corpus.split[set]:
            sent_examples = []
            previous_arcs = []
            stack = [Word(0, 'root', 'root', 'root', -1, '')]
            buffer = [word for word in corpus.data_set[sent_id]]
            configuration = Configuration(stack, buffer, previous_arcs)
            self.training_dynamic_oracle_sentence(configuration, sent_examples, sent_id, corpus.gold_trees[sent_id].list_of_arcs)
            examples[sent_id] = sent_examples
        return examples


    def training_dynamic_oracle_sentence(self, configuration, examples, sent_id, gold_arcs):
        b = configuration.buffer[0]
        s = configuration.stack[-1]
        right_arc = self.transition_name2transition[f"right_arc{self.use_labels*('_'+b.deprel)}"]
        left_arc = self.transition_name2transition[f"left_arc{self.use_labels*('_'+s.deprel)}"]
        reduce = self.transition_name2transition["reduce"]
        shift = self.transition_name2transition["shift"]
        done = self.transition_name2transition["done"]

        if right_arc.check(configuration):
            if right_arc.cost(configuration, gold_arcs) == 0:
                examples.append(Example(configuration, right_arc))
                new_configuration = configuration.copy()
                right_arc.apply(new_configuration)
                self.training_dynamic_oracle_sentence(new_configuration, examples, sent_id, gold_arcs)

        if left_arc.check(configuration):
            if left_arc.cost(configuration, gold_arcs) == 0:
                examples.append(Example(configuration, left_arc))
                new_configuration = configuration.copy()
                left_arc.apply(new_configuration)
                self.training_dynamic_oracle_sentence(new_configuration, examples, sent_id, gold_arcs)

        if shift.check(configuration):
            if shift.cost(configuration, gold_arcs) == 0:
                examples.append(Example(configuration, shift))
                new_configuration = configuration.copy()
                shift.apply(new_configuration)
                self.training_dynamic_oracle_sentence(new_configuration, examples, sent_id, gold_arcs)

        if reduce.check(configuration):
            if reduce.cost(configuration, gold_arcs) == 0:
                examples.append(Example(configuration, reduce))
                new_configuration = configuration.copy()
                reduce.apply(new_configuration)
                self.training_dynamic_oracle_sentence(new_configuration, examples, sent_id, gold_arcs)

        if done.check(configuration):
            examples.append(Example(configuration, done))


def loss_graph(arcs, gold_arcs):
    # silence = [arc for arc in gold_arcs if arc not in arcs]
    same = 0
    for gold_arc in gold_arcs:
        for arc in arcs:
            if arc == gold_arc:
                same += 1
    return len(gold_arcs) - same