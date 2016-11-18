#!/usr/bin/env python2.7

from __future__ import print_function

desc = '''Generate HTML output to visualize optimization records from the YAML files
generated with -fsave-optimization-record and -fdiagnostics-show-hotness.

The tools requires PyYAML to be installed.'''

import yaml
import argparse
import os.path
import subprocess
import shutil

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('yaml_files', nargs='+')
parser.add_argument('output_dir')
args = parser.parse_args()

p = subprocess.Popen(['c++filt', '-n'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def demangle(name):
    p.stdin.write(name + '\n')
    return p.stdout.readline().rstrip()

class Remark(yaml.YAMLObject):
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
        return "{}#L{}".format(SourceFileRenderer.html_file_name(self.File), self.Line)

    def getArgString(self, pair):
        if pair[0] == 'Callee' or pair[0] == 'Caller':
            return demangle(pair[1])
        return pair[1]

    @property
    def message(self):
        # Args is a list of mappings (dictionaries) with each dictionary with
        # exactly one key-value pair.
        values = [self.getArgString(mapping.items()[0]) for mapping in self.Args]
        return demangle("".join(values))

class Analysis(Remark):
    yaml_tag = '!Analysis'

    @property
    def color(self): return "white"

class AnalysisFPCommute(Analysis):
    yaml_tag = '!AnalysisFPCommute'

class AnalysisAliasing(Analysis):
    yaml_tag = '!AnalysisAliasing'

class Passed(Remark):
    yaml_tag = '!Passed'

    @property
    def color(self): return "green"

class Missed(Remark):
    yaml_tag = '!Missed'

    @property
    def color(self): return "red"

class SourceFileRenderer:
    def __init__(self, filename):
        self.source_stream = open(filename)
        self.stream = open(os.path.join(args.output_dir, SourceFileRenderer.html_file_name(filename)), 'w')

    def render_source_line(self, linenum, line):
        print('''
<tr>
<td><a name=\"L{linenum}\">{linenum}</a></td>
<td></td>
<td></td>
<td><pre>{line}</pre></td>
</tr>'''.format(**locals()), file=self.stream)

    def render_inline_remarks(self, r):
        print('''
<tr>
<td></td>
<td>{r.Hotness}</td>
<td class=\"column-entry-{r.color}\">{r.Pass}</td>
<td class=\"column-entry-yellow\">{r.message}</td>
</tr>'''.format(**locals()), file=self.stream)

    def render(self, line_remarks):
        print('''
<html>
<head>
<link rel='stylesheet' type='text/css' href='style.css'>
</head>
<body>
<div class="centered">
<table>
<tr>
<td>Line</td>
<td>Hotness</td>
<td>Optimization</td>
<td>Source</td>
</tr>''', file=self.stream)
        for (linenum, line) in enumerate(self.source_stream.readlines(), start=1):
            self.render_source_line(linenum, line)
            for remark in line_remarks.get(linenum, []):
                self.render_inline_remarks(remark)
        print('''
</table>
</body>
</html>''', file=self.stream)

    @classmethod
    def html_file_name(cls, filename):
        return filename.replace('/', '_') + ".html"

class IndexRenderer:
    def __init__(self):
        self.stream = open(os.path.join(args.output_dir, 'index.html'), 'w')

    def render_entry(self, r):
        print('''
<tr>
<td><a href={r.Link}>{r.DebugLocString}</a></td>
<td>{r.Hotness}%</td>
<td>{r.DemangledFunctionName}</td>
<td class=\"column-entry-{r.color}\">{r.Pass}</td>
</tr>'''.format(**locals()), file=self.stream)

    def render(self, all_remarks):
        print('''
<html>
<head>
<link rel='stylesheet' type='text/css' href='style.css'>
</head>
<body>
<div class="centered">
<table>
<tr>
<td>Source Location</td>
<td>Hotness</td>
<td>Function</td>
<td>Pass</td>
</tr>''', file=self.stream)
        for remark in all_remarks:
            self.render_entry(remark)
        print('''
</table>
</body>
</html>''', file=self.stream)


all_remarks = []
file_remarks  = dict()

for input_file in args.yaml_files:
    f = open(input_file)
    docs = yaml.load_all(f)
    for remark in docs:
        if hasattr(remark, 'Hotness'):
            file_remarks.setdefault(remark.File, dict()).setdefault(remark.Line, []).append(remark);
            all_remarks.append(remark)

all_remarks = sorted(all_remarks, key=lambda r: r.Hotness, reverse=True)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

for (filename, remarks) in file_remarks.iteritems():
    SourceFileRenderer(filename).render(remarks)

IndexRenderer().render(all_remarks)

shutil.copy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "style.css"), args.output_dir)
