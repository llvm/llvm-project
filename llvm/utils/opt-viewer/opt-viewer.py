#!/usr/bin/env python2.7

from __future__ import print_function

desc = '''Generate HTML output to visualize optimization records from the YAML files
generated with -fsave-optimization-record and -fdiagnostics-show-hotness.

The tools requires PyYAML and Pygments Python packages.'''

import optrecord
import functools
from multiprocessing import Pool
from multiprocessing import Lock, cpu_count
import errno
import argparse
import os.path
import re
import shutil
from pygments import highlight
from pygments.lexers.c_cpp import CppLexer
from pygments.formatters import HtmlFormatter
import cgi

# This allows passing the global context to the child processes.
class Context:
    def __init__(self, caller_loc = dict()):
       # Map function names to their source location for function where inlining happened
       self.caller_loc = caller_loc

context = Context()

class SourceFileRenderer:
    def __init__(self, source_dir, output_dir, filename):
        existing_filename = None
        if os.path.exists(filename):
            existing_filename = filename
        else:
            fn = os.path.join(source_dir, filename)
            if os.path.exists(fn):
                existing_filename = fn

        self.stream = open(os.path.join(output_dir, optrecord.html_file_name(filename)), 'w')
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
        self.cpp_lexer = CppLexer(stripnl=False)

    def render_source_lines(self, stream, line_remarks):
        file_text = stream.read()
        html_highlighted = highlight(file_text, self.cpp_lexer, self.html_formatter)

        # Take off the header and footer, these must be
        #   reapplied line-wise, within the page structure
        html_highlighted = html_highlighted.replace('<div class="highlight"><pre>', '')
        html_highlighted = html_highlighted.replace('</pre></div>', '')

        for (linenum, html_line) in enumerate(html_highlighted.split('\n'), start=1):
            print('''
<tr>
<td><a name=\"L{linenum}\">{linenum}</a></td>
<td></td>
<td></td>
<td><div class="highlight"><pre>{html_line}</pre></div></td>
</tr>'''.format(**locals()), file=self.stream)

            for remark in line_remarks.get(linenum, []):
                self.render_inline_remarks(remark, html_line)

    def render_inline_remarks(self, r, line):
        inlining_context = r.DemangledFunctionName
        dl = context.caller_loc.get(r.Function)
        if dl:
            link = optrecord.make_link(dl['File'], dl['Line'] - 2)
            inlining_context = "<a href={link}>{r.DemangledFunctionName}</a>".format(**locals())

        # Column is the number of characters *including* tabs, keep those and
        # replace everything else with spaces.
        indent = line[:max(r.Column, 1) - 1]
        indent = re.sub('\S', ' ', indent)

        print('''
<tr>
<td></td>
<td>{r.RelativeHotness}</td>
<td class=\"column-entry-{r.color}\">{r.PassWithDiffPrefix}</td>
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
        self.render_source_lines(self.source_stream, line_remarks)

        print('''
</table>
</body>
</html>''', file=self.stream)


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
<td class=\"column-entry-{r.color}\">{r.PassWithDiffPrefix}</td>
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


def _render_file(source_dir, output_dir, ctx, entry):
    global context
    context = ctx
    filename, remarks = entry
    SourceFileRenderer(source_dir, output_dir, filename).render(remarks)


def map_remarks(all_remarks):
    # Set up a map between function names and their source location for
    # function where inlining happened
    for remark in all_remarks.itervalues():
        if isinstance(remark, optrecord.Passed) and remark.Pass == "inline" and remark.Name == "Inlined":
            for arg in remark.Args:
                caller = arg.get('Caller')
                if caller:
                    context.caller_loc[caller] = arg['DebugLoc']


def generate_report(pmap, all_remarks, file_remarks, source_dir, output_dir, should_display_hotness):
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise

    _render_file_bound = functools.partial(_render_file, source_dir, output_dir, context)
    pmap(_render_file_bound, file_remarks.items())

    if should_display_hotness:
        sorted_remarks = sorted(all_remarks.itervalues(), key=lambda r: (r.Hotness, r.File, r.Line, r.Column, r.__dict__), reverse=True)
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
        help='Max job count (defaults to %(default)s, the current CPU count)')
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

    all_remarks, file_remarks, should_display_hotness = optrecord.gather_results(pmap, args.yaml_files)

    map_remarks(all_remarks)

    generate_report(pmap, all_remarks, file_remarks, args.source_dir, args.output_dir, should_display_hotness)
