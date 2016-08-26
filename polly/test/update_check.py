#! /usr/bin/env python3
# -*- coding: UTF-8 -*-

# Polly/LLVM update_check.py
# Update lit FileCheck files by replacing the 'CHECK:' lines by the actual output of the 'RUN:' command.

import argparse
import os
import subprocess
import shlex
import re


polly_src_dir = '''@POLLY_SOURCE_DIR@'''
polly_lib_dir = '''@POLLY_LIB_DIR@'''
shlibext = '''@LLVM_SHLIBEXT@'''
llvm_tools_dir = '''@LLVM_TOOLS_DIR@'''
link_polly_into_tools = not '''@LINK_POLLY_INTO_TOOLS@'''.lower() in {'','0','n','no','off','false','notfound','link_polly_into_tools-notfound'}

runre = re.compile(r'\s*\;\s*RUN\s*\:(?P<tool>.*)')
filecheckre = re.compile(r'\s*(?P<tool>.*)\|\s*(?P<filecheck>FileCheck\s[^|]*)')
emptyline = re.compile(r'\s*(\;\s*)?')
commentline = re.compile(r'\s*(\;.*)?')


def ltrim_emptylines(lines,meta=None):
    while len(lines) and emptyline.fullmatch(lines[0]):
        del lines[0]
        if meta is not None:
            del meta[0]


def rtrim_emptylines(lines):
    while len(lines) and emptyline.fullmatch(lines[-1]):
        del lines[-1]


def trim_emptylines(lines):
    ltrim_emptylines(lines)
    rtrim_emptylines(lines)


def complete_exename(path, filename):
    complpath = os.path.join(path, filename)
    if os.path.isfile(complpath):
        return complpath
    elif os.path.isfile(complpath + '.exe'):
        return complpath + '.exe'
    return filename


def indention(line):
    for i,c in enumerate(line):
        if c != ' ' and c != '\t':
            return i
    return None


def common_indent(lines):
    indentions = (indention(line) for line in lines)
    indentions = (indent for indent in indentions if indent is not None)
    return min(indentions,default=0)


funcre = re.compile(r'^    Function: \S*$')
regionre = re.compile(r'^    Region: \S*$')
depthre = re.compile(r'^    Max Loop Depth: .*')
paramre = re.compile(r'    [0-9a-z-A-Z_]+\: .*')

def classyfier1(lines):
    i = iter(lines)
    line = i.__next__()
    while True:
        if line.startswith("Printing analysis 'Polly - Calculate dependences' for region: "):
            yield {'PrintingDependenceInfo'}
        elif line.startswith("remark: "):
            yield {'Remark'}
        elif funcre.fullmatch(line):
            yield {'Function'}
        elif regionre.fullmatch(line):
            yield  { 'Region'}
        elif depthre.fullmatch(line):
            yield  {'MaxLoopDepth'}
        elif line == '    Invariant Accesses: {':
            while True:
                yield { 'InvariantAccesses'}
                if line == '    }':
                    break
                line = i.__next__()
        elif line == '    Context:':
            yield  {'Context'}
            line = i.__next__()
            yield  {'Context'}
        elif line == '    Assumed Context:':
            yield  {'AssumedContext'}
            line = i.__next__()
            yield  {'AssumedContext'}
        elif line == '    Invalid Context:':
            yield  {'InvalidContext'}
            line = i.__next__()
            yield  {'InvalidContext'}
        elif line == '    Boundary Context:':
            yield  {'BoundaryContext'}
            line = i.__next__()
            yield  {'BoundaryContext'}
            line = i.__next__()
            while paramre.fullmatch(line):
                yield  {'Param'}
                line = i.__next__()
            continue
        elif line == '    Arrays {':
            while True:
                yield  {'Arrays'}
                if line == '    }':
                    break
                line = i.__next__()
        elif line == '    Arrays (Bounds as pw_affs) {':
            while True:
                yield  {'PwAffArrays'}
                if line == '    }':
                    break
                line = i.__next__()
        elif line.startswith('    Alias Groups ('):
            while True:
                yield  {'AliasGroups'}
                line = i.__next__()
                if not line.startswith('        '):
                    break
            continue
        elif line == '    Statements {':
            while True:
                yield  {'Statements'}
                if line == '    }':
                    break
                line = i.__next__()
        elif line == '    RAW dependences:':
            yield {'RAWDep','BasicDep','Dep','DepInfo'}
            line = i.__next__()
            while line.startswith('        '):
                yield  {'RAWDep','BasicDep','Dep','DepInfo'}
                line = i.__next__()
            continue
        elif line == '    WAR dependences:':
            yield {'WARDep','BasicDep','Dep','DepInfo'}
            line = i.__next__()
            while line.startswith('        '):
                yield  {'WARDep','BasicDep','Dep','DepInfo'}
                line = i.__next__()
            continue
        elif line == '    WAW dependences:':
            yield {'WAWDep','BasicDep','Dep','DepInfo'}
            line = i.__next__()
            while line.startswith('        '):
                yield  {'WAWDep','BasicDep','Dep','DepInfo'}
                line = i.__next__()
            continue
        elif line == '    Reduction dependences:':
            yield {'RedDep','Dep','DepInfo'}
            line = i.__next__()
            while line.startswith('        '):
                yield  {'RedDep','Dep','DepInfo'}
                line = i.__next__()
            continue
        elif line == '    Transitive closure of reduction dependences:':
            yield {'TransitiveClosureDep','DepInfo'}
            line = i.__next__()
            while line.startswith('        '):
                yield  {'TransitiveClosureDep','DepInfo'}
                line = i.__next__()
            continue
        elif line.startswith("New access function '"):
            yield {'NewAccessFunction'}
        else:
            yield set()
        line = i.__next__()


def classyfier2(lines):
    i = iter(lines)
    line = i.__next__()
    while True:
        if funcre.fullmatch(line):
            while line.startswith('    '):
                yield  {'FunctionDetail'}
                line = i.__next__()
            continue
        elif line.startswith("Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: "):
            yield {'PrintingIslAst'}
            line = i.__next__()
            while not line.startswith('Printing analysis'):
                yield  {'AstDetail'}
                line = i.__next__()
            continue
        else:
            yield set()
        line = i.__next__()


replrepl = {'{{':'{{[{][{]}}','}}': '{{[}][}]}}', '[[':'{{\[\[}}',']]': '{{\]\]}}'}
replre = re.compile('|'.join(re.escape(k) for k in replrepl.keys()))

def main():
    parser = argparse.ArgumentParser(description="Update CHECK lines")
    parser.add_argument('testfile',help="File to update (absolute or relative to --testdir)")
    parser.add_argument('--check-style',choices=['CHECK','CHECK-NEXT'],default='CHECK-NEXT',help="What kind of checks lines to generate")
    parser.add_argument('--check-position',choices=['end','before-content','autodetect'],default='autodetect',help="Where to add the CHECK lines into the file; 'autodetect' searches for the first 'CHECK' line ind inserts it there")
    parser.add_argument('--check-include',action='append',default=[], help="What parts of the output lines to check; use syntax 'CHECK=include' to apply to one CHECK-prefix only (by default, everything)")
    parser.add_argument('--check-label-include',action='append',default=[],help="Use CHECK-LABEL for these includes")
    parser.add_argument('--check-part-newline',action='store_true',help="Add empty line between different check parts")
    parser.add_argument('--prefix-only',action='append',default=None,help="Update only these prefixes (default: all)")
    parser.add_argument('--bindir',help="Location of the opt program")
    parser.add_argument('--testdir',help="Root dir for unit tests")
    parser.add_argument('--inplace','-i',action='store_true',help="Replace input file")
    parser.add_argument('--output','-o',help="Write changed input to this file")
    known = parser.parse_args()

    if not known.inplace and known.output is None:
        print("Must specify what to do with output (--output or --inplace)")
        exit(1)
    if known.inplace and known.output is not None:
        print("--inplace and --output are mutually exclusive")
        exit(1)

    outfile = known.output

    filecheckparser = argparse.ArgumentParser(add_help=False)
    filecheckparser.add_argument('-check-prefix','--check-prefix',default='CHECK')

    filename = known.testfile
    for dir in ['.', known.testdir, os.path.join(polly_src_dir,'test'), polly_src_dir]:
        if not dir:
            continue
        testfilename = os.path.join(dir,filename)
        if os.path.isfile(testfilename):
            filename = testfilename
            break

    if known.inplace:
        outfile = filename

    allchecklines = []
    checkprefixes = []

    with open(filename, 'r') as file:
        oldlines = [line.rstrip('\r\n') for line in file.readlines()]

    runlines = []
    for line in oldlines:
        m = runre.match(line)
        if m:
            runlines.append(m.group('tool'))

    continuation = ''
    newrunlines = []
    for line in runlines:
        if line.endswith('\\'):
            continuation += line[:-2] + ' '
        else:
            newrunlines.append(continuation + line)
            continuation = ''
    if continuation:
        newrunlines.append(continuation)

    for line in newrunlines:
        m = filecheckre.match(line)
        if not m:
            continue

        tool, filecheck = m.group('tool', 'filecheck')
        filecheck = shlex.split(filecheck)
        tool = shlex.split(tool)
        if known.bindir is not None:
            tool[0] = complete_exename(known.bindir, tool[0])
        if os.path.isdir(llvm_tools_dir):
            tool[0] = complete_exename(llvm_tools_dir, tool[0])
        check_prefix = filecheckparser.parse_known_args(filecheck)[0].check_prefix
        if known.prefix_only is not None and not check_prefix in known.prefix_only:
            continue
        if check_prefix in checkprefixes:
            continue
        checkprefixes.append(check_prefix)

        newtool = []
        optstderr = None
        for toolarg in tool:
            toolarg = toolarg.replace('%s', filename)
            toolarg = toolarg.replace('%S', os.path.dirname(filename))
            if toolarg == '%loadPolly':
                if not link_polly_into_tools:
                    newtool += ['-load',os.path.join(polly_lib_dir,'LLVMPolly' + shlibext)]
                newtool.append('-polly-process-unprofitable')
                newtool.append('-polly-remarks-minimal')
            elif toolarg == '2>&1':
                optstderr = subprocess.STDOUT
            else:
                newtool.append(toolarg)
        tool = newtool

        inpfile = None
        i = 1
        while i <  len(tool):
            if tool[i] == '<':
                inpfile = tool[i + 1]
                del tool[i:i + 2]
                continue
            i += 1
        if inpfile:
            with open(inpfile) as inp:
                retlines = subprocess.check_output(tool,universal_newlines=True,stdin=inp,stderr=optstderr)
        else:
            retlines = subprocess.check_output(tool,universal_newlines=True,stderr=optstderr)
        retlines = [line.replace('\t', '    ') for line in retlines.splitlines()]
        check_include = []
        for checkme in known.check_include + known.check_label_include:
            parts = checkme.split('=')
            if len(parts) == 2:
                if parts[0] == check_prefix:
                    check_include.append(parts[1])
            else:
                check_include.append(checkme)

        if check_include:
            filtered_retlines = []
            classified_retlines = []
            lastmatch = None
            for line,kind in ((line,class1.union(class2)) for line,class1,class2 in zip(retlines,classyfier1(retlines), classyfier2(retlines))):
                match = kind.intersection(check_include)
                if match:
                    if lastmatch != match:
                        filtered_retlines.append('')
                        classified_retlines.append({'Separator'})
                    filtered_retlines.append(line)
                    classified_retlines.append(kind)
                lastmatch = match

            retlines = filtered_retlines
        else:
            classified_retlines = (set() for line in retlines)

        rtrim_emptylines(retlines)
        ltrim_emptylines(retlines,classified_retlines)
        retlines = [replre.sub(lambda m: replrepl[m.group(0)], line) for line in retlines]
        indent = common_indent(retlines)
        retlines = [line[indent:] for line in retlines]
        checklines = []
        previous_was_empty = True
        for line,kind in zip(retlines,classified_retlines):
            if line:
                if known.check_style == 'CHECK' and known.check_label_include:
                    if not kind.isdisjoint(known.check_label_include):
                        checklines.append('; ' + check_prefix + '-LABEL: ' + line)
                    else:
                        checklines.append('; ' + check_prefix + ':       ' + line)
                elif known.check_style == 'CHECK':
                    checklines.append('; ' + check_prefix + ': ' + line)
                elif known.check_label_include and known.check_label_include:
                    if not kind.isdisjoint(known.check_label_include):
                        checklines.append('; ' + check_prefix + '-LABEL: ' + line)
                    elif previous_was_empty:
                        checklines.append('; ' + check_prefix + ':       ' + line)
                    else:
                        checklines.append('; ' + check_prefix + '-NEXT:  ' + line)
                else:
                    if previous_was_empty:
                        checklines.append('; ' + check_prefix + ':      ' + line)
                    else:
                        checklines.append('; ' + check_prefix + '-NEXT: ' + line)
                previous_was_empty = False
            else:
                if not 'Separator' in kind or known.check_part_newline:
                    checklines.append(';')
                previous_was_empty = True
        allchecklines.append(checklines)

    if not checkprefixes:
        return

    checkre = re.compile(r'^\s*\;\s*(' + '|'.join([re.escape(s) for s in checkprefixes]) + ')(\-NEXT|\-DAG|\-NOT|\-LABEL|\-SAME)?\s*\:')
    firstcheckline = None
    firstnoncommentline = None
    headerlines = []
    newlines = []
    uptonowlines = []
    emptylines = []
    lastwascheck = False
    for line in oldlines:
        if checkre.match(line):
            if firstcheckline is None:
                firstcheckline = len(newlines) + len(emptylines)
            if not lastwascheck:
                uptonowlines += emptylines
            emptylines = []
            lastwascheck = True
        elif emptyline.fullmatch(line):
            emptylines.append(line)
        else:
            newlines += uptonowlines
            newlines += emptylines
            newlines.append(line)
            emptylines = []
            uptonowlines = []
            lastwascheck = False

    for i,line in enumerate(newlines):
        if not commentline.fullmatch(line):
            firstnoncommentline = i
            break

    with open(outfile,'w',newline='') as file:
        def writelines(lines):
            for line in lines:
                file.write(line)
                file.write('\n')

        if firstcheckline is not None and known.check_position == 'autodetect':
            writelines(newlines[:firstcheckline])
            writelines(uptonowlines)
            for i,checklines in enumerate(allchecklines):
                if i != 0:
                    file.write('\n')
                writelines(checklines)
            writelines(newlines[firstcheckline:])
            writelines(emptylines)
        elif firstnoncommentline is not None and known.check_position == 'before-content':
            headerlines = newlines[:firstnoncommentline]
            rtrim_emptylines(headerlines)
            contentlines = newlines[firstnoncommentline:]
            ltrim_emptylines(contentlines)

            writelines(headerlines)
            for checklines in allchecklines:
                file.write('\n')
                writelines(checklines)
            file.write('\n')
            writelines(contentlines)
            writelines(uptonowlines)
            writelines(emptylines)
        else:
            writelines(newlines)
            rtrim_emptylines(newlines)
            for checklines in allchecklines:
                file.write('\n\n')
                writelines(checklines)


if __name__ == '__main__':
    main()
