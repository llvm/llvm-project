#!/usr/bin/env python
##
## Name:    mkdoc.py
## Purpose: Extract documentation from header files.
##
## Copyright (C) 2018 Michael J. Fromberger. All Rights Reserved.
##
## Usage: mkdoc.py <template> <output>
##
from __future__ import print_function

import collections, re, sys

# A regular expression to match commented declarations.
# This is specific to C and not very general; it should work fine for the imath
# headers but will not adapt well to arbitrary code or to C++.
doc = re.compile(r'''(?mx)/\*\* # open  /**
(?P<text>(?:[^*]|\*[^/])*)      # text      Does a thing
\*/\n                           # close */
(?P<decl>[^;{]*(?:;$|\{))''')  # decl  void f(x);

# A regular expression matching up to 4 spaces at the head of a line.
spc = re.compile(r'(?m)^ {1,4}')

# A regular expression matching an insertion point.  An insertion point has the
# form {{include "header" name ...}}.  If no names are given, all the names in
# the given header are inserted.
ins = re.compile(r'{{insert "(?P<file>[^"]*)"(?P<names>(?:\s+\w+)+)?\s*}}')

# A regular expression matching non-identifier characters, for splitting.
nid = re.compile(r'\W+')

# A cache of already-parsed files, maps filename to declarations.
CACHE = {}


def last_word(s):
    """Returns the last identifier-shaped word in s."""
    return nid.split(s.strip())[-1]


def typeset(text):
    """Renders text with verbatim sections into markdown."""
    lines = []
    fence = False
    for line in text.split('\n'):
        if fence != line.startswith(' '):
            lines.append('```')
            fence = not fence
        lines.append(line)
    if fence:
        lines.append('```')
    for i, line in enumerate(lines):
        if i == 0: lines[i] = ' -  ' + line
        elif line: lines[i] = '    ' + line
    return '\n'.join(lines)


class LIndex(object):
    """Represents a line offset index for text."""

    def __init__(self, text):
        pos = 0

        # An array of ending offsets for each line, with a sentinel at position
        # 0 to make the index arithmetic easier.
        idx = [0]

        # Scan forward for newlines or EOF, and push the offsets of the line
        # breaks onto the list so we can binary search them later.
        while pos < len(text):
            next = text.find('\n', pos)
            if next < 0:
                break
            idx.append(next)
            pos = next + 1
        if idx[-1] < len(text):
            idx.append(len(text))
        self._len = len(text)
        self._index = idx

    def linecol(self, pos):
        """Returns the (line, col) corresponding to pos.

        Line numbers are 1-based, columns are 0-based.
        """
        if pos < 0 or pos > self._len:
            raise IndexError("position %d out of range" % pos)

        # Binary search for the largest line number whose end marker is at or
        # after pos and whose previous line's end is before pos.
        idx = self._index
        i, j = 1, len(idx)
        while i < j:
            m = (i + j) / 2
            if idx[m] < pos:
                i = m + 1
            elif idx[m - 1] < pos:
                return m, pos - idx[m - 1]
            else:
                j = m

        # This happens if (and only if) the whole file is one line.
        return 1, pos


class Decl(object):
    """Represents a single documented declaration."""

    def __init__(self, com, decl, line=None):
        """Initialize a new documented declaration.

        Params:
          com: the raw text of the comment
          decl: the raw text of the declaration
          line: the line number of the declaration
        """
        lp = decl.find('(')
        if lp < 0:
            self.name = last_word(decl.rstrip(';'))
        else:
            self.name = last_word(decl[:lp])
        self.decl = ' '.join(decl.rstrip(';{').strip().split())
        self.comment = spc.sub('', com.rstrip())
        self.line = line

    def __repr__(self):
        return '#Decl["%s"]' % self.decl

    def markdown(self, path):
        pos = self.decl.index(self.name)
        decl = '%s<a href="%s#L%d">%s</a>%s' % (
            self.decl[:pos],
            path,
            self.line,
            self.name,
            self.decl[pos + len(self.name):],
        )
        return '''------------
<a id="{name}"></a><pre>
{decl};
</pre>
{comment}
'''.format(name=self.name, decl=decl, comment=typeset(self.comment))


def parse_decls(text):
    """Parse a dictionary of declarations from text."""
    decls = collections.OrderedDict()
    idx = LIndex(text)
    for m in doc.finditer(text):
        line, _ = idx.linecol(m.span('decl')[0])
        d = Decl(m.group('text'), m.group('decl'), line)
        decls[d.name] = d
    return decls


def load_file(path):
    """Load declarations from path, or use cached results."""
    if path not in CACHE:
        with file(path, 'rU') as fp:
            CACHE[path] = parse_decls(fp.read())
    return CACHE[path]


def main(args):
    if len(args) != 2:
        print("Usage: mkdoc.py <input> <output>", file=sys.stderr)
        sys.exit(1)

    doc_template = args[0]
    doc_markdown = args[1]

    with file(doc_template, 'rU') as input:
        template = input.read()

    with file(doc_markdown, 'wt') as output:
        print(
            '''<!--
  This file was generated from "{0}" by mkdoc.py
  DO NOT EDIT
-->
'''.format(doc_template),
            file=output)

        pos = 0  # last position of input copied

        # Look for substitution markers in the template, and replace them with
        # their content.
        for ip in ins.finditer(template):
            output.write(template[pos:ip.start()])
            pos = ip.end()

            decls = load_file(ip.group('file'))
            if ip.group('names'):  # pick the selected names, in order
                decls = collections.OrderedDict(
                    (key, decls[key])
                    for key in ip.group('names').strip().split())

            # Render the selected declarations.
            for decl in decls.values():
                print(decl.markdown(ip.group('file')), file=output)

        # Clean up any remaining template bits
        output.write(template[pos:])


if __name__ == "__main__":
    main(sys.argv[1:])
