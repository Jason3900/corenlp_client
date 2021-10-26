# -*- coding:UTF-8 -*-
import requests
import warnings
import os
import re
from nltk import Tree
from subprocess import Popen
import subprocess
import time
import shlex
import multiprocessing
from urllib import parse

class CoreNLP:
    def __init__(self, url=None, lang="en", annotators=None, corenlp_dir=None, local_port=9000, max_mem=4, threads=multiprocessing.cpu_count(), timeout=150000):
        if url:
            self.url = url.rstrip("/")
        self.annotators_list = ["tokenize","ssplit","pos","ner","parse","depparse","openie"]
        self.lang = lang
        self.corenlp_subprocess = None
        self.timeout = timeout
        if annotators and self._check_annotators_format(annotators):
            self.annotators = annotators
        else:
            self.annotators = ",".join(self.annotators_list)
        
        if corenlp_dir:
            try:
                os.path.exists(corenlp_dir)
            except:
                raise OSError("please check corenlp local path is correct! ")
            if self._launch_local_server(corenlp_dir, local_port, max_mem, threads):
                self.url = f"http://127.0.0.1:{local_port}"
                self._request_corenlp(data="", annotators=self.annotators)
        
    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        if self.corenlp_subprocess:
            self.corenlp_subprocess.kill()
            self.corenlp_subprocess.wait()
        # os.killpg(os.getpgid(self.corenlp_subprocess.pid), 9)

    def __del__(self):
        if self.corenlp_subprocess:
            self.corenlp_subprocess.kill()
            self.corenlp_subprocess.wait()
    
    def _check_annotators_format(self, annotators):
        annotators = annotators.split(",")
        for i in annotators:
            if i not in self.annotators_list:
                return False
        return True

    def _check_server_status(self):
        if requests.get(self.url, verify=False).status_code != 200:
            raise ConnectionError("please check your network connection, or the corenlp server is started before launching!")
    
    @staticmethod
    def _deal_path_suffix(path):
        if "\\" in path:
            path = path.rstrip("\\") + "\\"
        else:
            path = path.rstrip("/") + "/"
        return path

    def _launch_local_server(self, corenlp_dir, port, max_mem, threads):
        corenlp_dir = self._deal_path_suffix(os.path.abspath(corenlp_dir))
        tmp_dir = "tmp"
        if not os.path.exists("tmp"):
            os.mkdir(tmp_dir)
        try:
            os.system("java -version")
        except:
            raise AssertionError("Java is required to launch corenlp server! ")
        cmd = f'java -Djava.io.tmpdir={tmp_dir} -mx{max_mem}g ' + \
            f'-cp "{corenlp_dir}*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer ' + \
                f'-threads {threads} -port {port} -timeout 150000 -lazy false'
        print(cmd)
        cmd = shlex.split(cmd)
        self.corenlp_subprocess = Popen(cmd)
        time.sleep(1)
        return True

    def _request_corenlp(self, data, annotators):
        params = {"properties": '{"annotators": "%s"}'  % annotators, "pipelineLanguage": self.lang}
        res = requests.post(url=self.url, params=params, data=parse.quote(data), timeout=self.timeout, verify=False)
        ann_result = res.json()
        return ann_result

    def annotate(self, data):
        ann_result = self._request_corenlp(data, self.annotators)
        annotation = Annotation(ann_result)
        return annotation

    def tokenize(self, data, ssplit=True):
        if ssplit:
            annotators = "tokenize,ssplit"
        else:
            annotators = "tokenize"
        ann_result = self._request_corenlp(data, annotators)
        if ssplit:
            annotation = [[token["word"] for token in sent["tokens"]] for sent in ann_result["sentences"]]
        else:
            annotation = [token["word"] for token in ann_result["tokens"]]
        return annotation

    def pos_tag(self, data):
        annotators = "tokenize,ssplit,pos"
        ann_result = self._request_corenlp(data, annotators)
        annotation = [[token["pos"] for token in sent["tokens"]] for sent in ann_result["sentences"]]
        return annotation

    def ner(self, data):
        annotators = "tokenize,ssplit,pos,ner"
        ann_result = self._request_corenlp(data, annotators)
        annotation = []
        for sent in ann_result["sentences"]:
            sent_ner = []
            if "entitymentions" in sent:
                for entity in sent["entitymentions"]:
                    span = (entity["characterOffsetBegin"], entity["characterOffsetEnd"])
                    ner = entity["ner"]
                    ner_entity = entity["text"]
                    sent_ner.append({(ner_entity,span): ner})
            annotation.append(sent_ner)
        return annotation

    @staticmethod
    def pretty_print_tree(tree):
        Tree.fromstring(tree).pretty_print()

    def close(self):
        if self.corenlp_subprocess:
            self.corenlp_subprocess.kill()
            self.corenlp_subprocess.wait()

class Annotation():
    def __init__(self, ann_result):
        self.ann_result = ann_result
        self.tokens=[]
        self.parse_tree=[]
        self.bi_parse_tree=[]
        self.basic_dep=[]
        self.enhanced_dep=[]
        self.enhanced_pp_dep=[]
        self.entities = []
        self.openie = []
        self._extract_ann()

    def _extract_ann(self):
        ann_dict = dict()
        if "sentences" in self.ann_result:
            for ann_sent in self.ann_result["sentences"]:
                self.tokens.append(ann_sent["tokens"])
                if "parse" in ann_sent:
                    self.parse_tree.append(re.sub(r"\s+", " ", ann_sent["parse"]))
                if "binaryParse" in ann_sent:
                    self.bi_parse_tree.append(re.sub(r"\s+", " ", ann_sent["binaryParse"]))
                if "basicDependencies" in ann_sent:
                    self.basic_dep.append(ann_sent["basicDependencies"])
                if "enhancedDependencies" in ann_sent:
                    self.enhanced_dep.append(ann_sent["enhancedDependencies"])
                if "enhancedPlusPlusDependencies" in ann_sent:
                    self.enhanced_pp_dep.append(ann_sent["enhancedPlusPlusDependencies"])
                if "entitymentions" in ann_sent:
                    self.entities.append(ann_sent["entitymentions"])
                if "openie" in ann_sent:
                    self.openie.append(ann_sent["openie"])
        else:
            self.tokens = self.ann_result["tokens"]
        return ann_dict







