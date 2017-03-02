#!/usr/bin/env python2.7

from __future__ import print_function

import yaml
# Try to use the C parser.
try:
    from yaml import CLoader as Loader
except ImportError:
    print("For faster parsing, you may want to install libYAML for PyYAML")
    from yaml import Loader

import functools
from collections import defaultdict
import itertools
from multiprocessing import Pool
from multiprocessing import Lock, cpu_count
import cgi
import subprocess

import traceback

p = subprocess.Popen(['c++filt', '-n'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
p_lock = Lock()


def demangle(name):
    with p_lock:
        p.stdin.write(name + '\n')
        return p.stdout.readline().rstrip()

def html_file_name(filename):
    return filename.replace('/', '_') + ".html"

def make_link(File, Line):
    return "{}#L{}".format(html_file_name(File), Line)


class Remark(yaml.YAMLObject):
    # Work-around for http://pyyaml.org/ticket/154.
    yaml_loader = Loader

    def __getattr__(self, name):
        # If hotness is missing, assume 0
        if name == 'Hotness':
            return 0
        raise AttributeError(name)

    @property
    def File(self):
        return self.DebugLoc['File']

    @property
    def Line(self):
        return int(self.DebugLoc['Line'])

    @property
    def Column(self):
        return self.DebugLoc['Column']

    @property
    def DebugLocString(self):
        return "{}:{}:{}".format(self.File, self.Line, self.Column)

    @property
    def DemangledFunctionName(self):
        return demangle(self.Function)

    @property
    def Link(self):
        return make_link(self.File, self.Line)

    def getArgString(self, mapping):
        mapping = mapping.copy()
        dl = mapping.get('DebugLoc')
        if dl:
            del mapping['DebugLoc']

        assert(len(mapping) == 1)
        (key, value) = mapping.items()[0]

        if key == 'Caller' or key == 'Callee':
            value = cgi.escape(demangle(value))

        if dl and key != 'Caller':
            return "<a href={}>{}</a>".format(
                make_link(dl['File'], dl['Line']), value)
        else:
            return value

    @property
    def message(self):
        # Args is a list of mappings (dictionaries)
        values = [self.getArgString(mapping) for mapping in self.Args]
        return "".join(values)

    @property
    def RelativeHotness(self):
        if self.max_hotness:
            return "{}%".format(int(round(self.Hotness * 100 / self.max_hotness)))
        else:
            return ''

    @property
    def key(self):
        return (self.__class__, self.Pass, self.Name, self.File, self.Line, self.Column, self.Function)


class Analysis(Remark):
    yaml_tag = '!Analysis'

    @property
    def color(self):
        return "white"


class AnalysisFPCommute(Analysis):
    yaml_tag = '!AnalysisFPCommute'


class AnalysisAliasing(Analysis):
    yaml_tag = '!AnalysisAliasing'


class Passed(Remark):
    yaml_tag = '!Passed'

    @property
    def color(self):
        return "green"


class Missed(Remark):
    yaml_tag = '!Missed'

    @property
    def color(self):
        return "red"


def get_remarks(input_file):
    max_hotness = 0
    all_remarks = dict()
    file_remarks = defaultdict(functools.partial(defaultdict, list))

    with open(input_file) as f:
        docs = yaml.load_all(f, Loader=Loader)
        for remark in docs:
            # Avoid remarks withoug debug location or if they are duplicated
            if not hasattr(remark, 'DebugLoc') or remark.key in all_remarks:
                continue
            all_remarks[remark.key] = remark

            file_remarks[remark.File][remark.Line].append(remark)

            max_hotness = max(max_hotness, remark.Hotness)

    return max_hotness, all_remarks, file_remarks


def gather_results(pmap, filenames):
    remarks = pmap(get_remarks, filenames)
    max_hotness = max(entry[0] for entry in remarks)

    def merge_file_remarks(file_remarks_job, all_remarks, merged):
        for filename, d in file_remarks_job.iteritems():
            for line, remarks in d.iteritems():
                for remark in remarks:
                    # Bring max_hotness into the remarks so that
                    # RelativeHotness does not depend on an external global.
                    remark.max_hotness = max_hotness
                    if remark.key not in all_remarks:
                        merged[filename][line].append(remark)

    all_remarks = dict()
    file_remarks = defaultdict(functools.partial(defaultdict, list))
    for _, all_remarks_job, file_remarks_job in remarks:
        merge_file_remarks(file_remarks_job, all_remarks, file_remarks)
        all_remarks.update(all_remarks_job)

    return all_remarks, file_remarks, max_hotness != 0
