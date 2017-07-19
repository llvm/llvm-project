#!/usr/bin/env python2.7

from __future__ import print_function

import yaml
# Try to use the C parser.
try:
    from yaml import CLoader as Loader
except ImportError:
    print("For faster parsing, you may want to install libYAML for PyYAML")
    from yaml import Loader

import cgi
from collections import defaultdict
import fnmatch
import functools
from multiprocessing import Lock
import os, os.path
import subprocess

import optpmap


p = subprocess.Popen(['c++filt', '-n'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
p_lock = Lock()


try:
    dict.iteritems
except AttributeError:
    # Python 3
    def itervalues(d):
        return iter(d.values())
    def iteritems(d):
        return iter(d.items())
else:
    # Python 2
    def itervalues(d):
        return d.itervalues()
    def iteritems(d):
        return d.iteritems()


def demangle(name):
    with p_lock:
        p.stdin.write((name + '\n').encode('utf-8'))
        p.stdin.flush()
        return p.stdout.readline().rstrip().decode('utf-8')


def html_file_name(filename):
    return filename.replace('/', '_').replace('#', '_') + ".html"


def make_link(File, Line):
    return "\"{}#L{}\"".format(html_file_name(File), Line)


class Remark(yaml.YAMLObject):
    # Work-around for http://pyyaml.org/ticket/154.
    yaml_loader = Loader

    def initmissing(self):
        if not hasattr(self, 'Hotness'):
            self.Hotness = 0
        if not hasattr(self, 'Args'):
            self.Args = []

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

    def getDiffPrefix(self):
        if hasattr(self, 'Added'):
            if self.Added:
                return '+'
            else:
                return '-'
        return ''

    @property
    def PassWithDiffPrefix(self):
        return self.getDiffPrefix() + self.Pass

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
        k = (self.__class__, self.PassWithDiffPrefix, self.Name, self.File, self.Line, self.Column, self.Function)
        for arg in self.Args:
            for (key, value) in iteritems(arg):
                if type(value) is dict:
                    value = tuple(value.items())
                k += (key, value)
        return k

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __repr__(self):
        return str(self.key)


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
            remark.initmissing()
            # Avoid remarks withoug debug location or if they are duplicated
            if not hasattr(remark, 'DebugLoc') or remark.key in all_remarks:
                continue
            all_remarks[remark.key] = remark

            file_remarks[remark.File][remark.Line].append(remark)

            # If we're reading a back a diff yaml file, max_hotness is already
            # captured which may actually be less than the max hotness found
            # in the file.
            if hasattr(remark, 'max_hotness'):
                max_hotness = remark.max_hotness
            max_hotness = max(max_hotness, remark.Hotness)

    return max_hotness, all_remarks, file_remarks


def gather_results(filenames, num_jobs, should_print_progress):
    if should_print_progress:
        print('Reading YAML files...')
    remarks = optpmap.pmap(
        get_remarks, filenames, num_jobs, should_print_progress)
    max_hotness = max(entry[0] for entry in remarks)

    def merge_file_remarks(file_remarks_job, all_remarks, merged):
        for filename, d in iteritems(file_remarks_job):
            for line, remarks in iteritems(d):
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


def find_opt_files(dirs_or_files):
    all = []
    for dir_or_file in dirs_or_files:
        if os.path.isfile(dir_or_file):
            all.append(dir_or_file)
        else:
            for dir, subdirs, files in os.walk(dir_or_file):
                # Exclude mounted directories and symlinks (os.walk default).
                subdirs[:] = [d for d in subdirs
                              if not os.path.ismount(os.path.join(dir, d))]
                for file in files:
                    if fnmatch.fnmatch(file, "*.opt.yaml"):
                        all.append(os.path.join(dir, file))
    return all
