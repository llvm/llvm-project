#!/usr/bin/env python2.7

from __future__ import print_function

desc = '''Generate HTML output to visualize optimization records from the YAML files
generated with -fsave-optimization-record and -fdiagnostics-show-hotness.

The tools requires PyYAML and Pygments Python packages.

For faster parsing, you may want to use libYAML with PyYAML.'''

import yaml
# Try to use the C parser.
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import functools
from collections import defaultdict
import itertools
from multiprocessing import Pool
from multiprocessing import Lock, cpu_count
import errno
import argparse
import os.path
import re
import subprocess
import shutil
from pygments import highlight
from pygments.lexers.c_cpp import CppLexer
from pygments.formatters import HtmlFormatter
import cgi

p = subprocess.Popen(['c++filt', '-n'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
p_lock = Lock()


def demangle(name):
    with p_lock:
        p.stdin.write(name + '\n')
        return p.stdout.readline().rstrip()

# This allows passing the global context to the child processes.
class Context:
    def __init__(self, max_hotness = 0, caller_loc = dict()):
       self.max_hotness = max_hotness

       # Map function names to their source location for function where inlining happened
       self.caller_loc = caller_loc

    def should_display_hotness(self):
        # If max_hotness is 0 at the end, we assume hotness information is
        # missing and no relative hotness information is displayed
        return self.max_hotness != 0

context = Context()

class Remark(yaml.YAMLObject):
    # Work-around for http://pyyaml.org/ticket/154.
    yaml_loader = Loader

    def __getattr__(self, name):
        # If hotness is missing, assume 0
        if name == 'Hotness':
            return 0
        raise AttributeError

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

    @classmethod
    def make_link(cls, File, Line):
        return "{}#L{}".format(SourceFileRenderer.html_file_name(File), Line)

    @property
    def Link(self):
        return Remark.make_link(self.File, self.Line)

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
                Remark.make_link(dl['File'], dl['Line']), value)
        else:
            return value

    @property
    def message(self):
        # Args is a list of mappings (dictionaries)
        values = [self.getArgString(mapping) for mapping in self.Args]
        return "".join(values)

    @property
    def RelativeHotness(self):
        if context.should_display_hotness():
            return "{}%".format(int(round(self.Hotness * 100 / context.max_hotness)))
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


class SourceFileRenderer:
    def __init__(self, source_dir, output_dir, filename):
        existing_filename = None
        if os.path.exists(filename):
            existing_filename = filename
        else:
            fn = os.path.join(source_dir, filename)
            if os.path.exists(fn):
                existing_filename = fn

        self.stream = open(os.path.join(output_dir, SourceFileRenderer.html_file_name(filename)), 'w')
        if existing_filename:
            self.source_stream = open(existing_filename)
        else:
            self.source_stream = None
            print('''
<html>
<h1>Unable to locate file {}</h1>
</html>
            '''.format(filename), file=self.stream)

        self.html_formatter = HtmlFormatter(encoding='utf-8')
        self.cpp_lexer = CppLexer()

    def render_source_line(self, linenum, line):
        html_line = highlight(line, self.cpp_lexer, self.html_formatter)
        print('''
<tr>
<td><a name=\"L{linenum}\">{linenum}</a></td>
<td></td>
<td></td>
<td>{html_line}</td>
</tr>'''.format(**locals()), file=self.stream)

    def render_inline_remarks(self, r, line):
        inlining_context = r.DemangledFunctionName
        print
        dl = context.caller_loc.get(r.Function)
        if dl:
            link = Remark.make_link(dl['File'], dl['Line'] - 2)
            inlining_context = "<a href={link}>{r.DemangledFunctionName}</a>".format(**locals())

        # Column is the number of characters *including* tabs, keep those and
        # replace everything else with spaces.
        indent = line[:r.Column - 1]
        indent = re.sub('\S', ' ', indent)

        print('''
<tr>
<td></td>
<td>{r.RelativeHotness}</td>
<td class=\"column-entry-{r.color}\">{r.Pass}</td>
<td><pre style="display:inline">{indent}</pre><span class=\"column-entry-yellow\"> {r.message}&nbsp;</span></td>
<td class=\"column-entry-yellow\">{inlining_context}</td>
</tr>'''.format(**locals()), file=self.stream)

    def render(self, line_remarks):
        if not self.source_stream:
            return

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
<td>Inline Context</td>
</tr>''', file=self.stream)
        for (linenum, line) in enumerate(self.source_stream.readlines(), start=1):
            self.render_source_line(linenum, line)
            for remark in line_remarks.get(linenum, []):
                self.render_inline_remarks(remark, line)
        print('''
</table>
</body>
</html>''', file=self.stream)

    @classmethod
    def html_file_name(cls, filename):
        return filename.replace('/', '_') + ".html"


class IndexRenderer:
    def __init__(self, output_dir):
        self.stream = open(os.path.join(output_dir, 'index.html'), 'w')

    def render_entry(self, r, odd):
        escaped_name = cgi.escape(r.DemangledFunctionName)
        print('''
<tr>
<td class=\"column-entry-{odd}\"><a href={r.Link}>{r.DebugLocString}</a></td>
<td class=\"column-entry-{odd}\">{r.RelativeHotness}</td>
<td class=\"column-entry-{odd}\">{escaped_name}</td>
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
        for i, remark in enumerate(all_remarks):
            self.render_entry(remark, i % 2)
        print('''
</table>
</body>
</html>''', file=self.stream)


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


def _render_file(source_dir, output_dir, ctx, entry):
    global context
    context = ctx
    filename, remarks = entry
    SourceFileRenderer(source_dir, output_dir, filename).render(remarks)


def gather_results(pmap, filenames):
    remarks = pmap(get_remarks, filenames)

    def merge_file_remarks(file_remarks_job, all_remarks, merged):
        for filename, d in file_remarks_job.iteritems():
            for line, remarks in d.iteritems():
                for remark in remarks:
                    if remark.key not in all_remarks:
                        merged[filename][line].append(remark)

    all_remarks = dict()
    file_remarks = defaultdict(functools.partial(defaultdict, list))
    for _, all_remarks_job, file_remarks_job in remarks:
        merge_file_remarks(file_remarks_job, all_remarks, file_remarks)
        all_remarks.update(all_remarks_job)

    context.max_hotness = max(entry[0] for entry in remarks)

    return all_remarks, file_remarks


def map_remarks(all_remarks):
    # Set up a map between function names and their source location for
    # function where inlining happened
    for remark in all_remarks.itervalues():
        if isinstance(remark, Passed) and remark.Pass == "inline" and remark.Name == "Inlined":
            for arg in remark.Args:
                caller = arg.get('Caller')
                if caller:
                    context.caller_loc[caller] = arg['DebugLoc']


def generate_report(pmap, all_remarks, file_remarks, source_dir, output_dir):
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise

    _render_file_bound = functools.partial(_render_file, source_dir, output_dir, context)
    pmap(_render_file_bound, file_remarks.items())

    if context.should_display_hotness():
        sorted_remarks = sorted(all_remarks.itervalues(), key=lambda r: (r.Hotness, r.__dict__), reverse=True)
    else:
        sorted_remarks = sorted(all_remarks.itervalues(), key=lambda r: (r.File, r.Line, r.Column, r.__dict__))
    IndexRenderer(args.output_dir).render(sorted_remarks)

    shutil.copy(os.path.join(os.path.dirname(os.path.realpath(__file__)),
            "style.css"), output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('yaml_files', nargs='+')
    parser.add_argument('output_dir')
    parser.add_argument(
        '--jobs',
        '-j',
        default=cpu_count(),
        type=int,
        help='Max job count (defaults to current CPU count)')
    parser.add_argument(
        '-source-dir',
        '-s',
        default='',
        help='set source directory')
    args = parser.parse_args()

    if len(args.yaml_files) == 0:
        parser.print_help()
        sys.exit(1)

    if args.jobs == 1:
        pmap = map
    else:
        pool = Pool(processes=args.jobs)
        pmap = pool.map

    all_remarks, file_remarks = gather_results(pmap, args.yaml_files)

    map_remarks(all_remarks)

    generate_report(pmap, all_remarks, file_remarks, args.source_dir, args.output_dir)
