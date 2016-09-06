#!/usr/bin/env python
#
# Given a previous good compile narrow down miscompiles.
# Expects two directories named "before" and "after" each containing a set of
# assembly or object files where the "after" version is assumed to be broken.
# You also have to provide a script called "link_test". It is called with a list
# of files which should be linked together and result tested. "link_test" should
# returns with exitcode 0 if the linking and testing succeeded.
#
# abtest.py operates by taking all files from the "before" directory and
# in each step replacing one of them with a file from the "bad" directory.
#
# Additionally you can perform the same steps with a single .s file. In this
# mode functions are identified by "# -- Begin FunctionName" and
# "# -- End FunctionName" markers. The abtest.py then takes all functions from
# the file in the "before" directory and replaces one function with the
# corresponding function from the "bad" file in each step.
#
# Example usage to identify miscompiled files:
#    1. Create a link_test script, make it executable. Simple Example:
#          clang "$@" -o /tmp/test && /tmp/test || echo "PROBLEM"
#    2. Run the script to figure out which files are miscompiled:
#       > ./abtest.py 
#       somefile.s: ok
#       someotherfile.s: skipped: same content
#       anotherfile.s: failed: './link_test' exitcode != 0
#       ...
# Example usage to identify miscompiled functions inside a file:
#    3. First you have to mark begin and end of the functions.
#       The script comes with some examples called mark_xxx.py.
#       Unfortunately this is very specific to your environment and it is likely
#       that you have to write a custom version for your environment.
#       > for i in before/*.s after/*.s; do mark_xxx.py $i; done
#    4. Run the tests on a single file (assuming before/file.s and
#       after/file.s exist)
#       > ./abtest.py file.s
#       funcname1 [0/XX]: ok
#       funcname2 [1/XX]: ok
#       funcname3 [2/XX]: skipped: same content
#       funcname4 [3/XX]: failed: './link_test' exitcode != 0
#       ...
from fnmatch import filter
from sys import stderr
import argparse
import filecmp
import os
import subprocess
import sys

LINKTEST="./link_test"
ESCAPE="\033[%sm"
BOLD=ESCAPE % "1"
RED=ESCAPE % "31"
NORMAL=ESCAPE % "0"
FAILED=RED+"failed"+NORMAL

def find(dir, file_filter=None):
    files = [walkdir[0]+"/"+file for walkdir in os.walk(dir) for file in walkdir[2]]
    if file_filter != None:
        files = filter(files, file_filter)
    return files

def error(message):
    stderr.write("Error: %s\n" % (message,))

def warn(message):
    stderr.write("Warning: %s\n" % (message,))

def extract_functions(file):
    functions = []
    in_function = None
    for line in open(file):
        if line.startswith("# -- Begin  "):
            if in_function != None:
                warn("Missing end of function %s" % (in_function,))
            funcname = line[12:-1]
            in_function = funcname
            text = line
        elif line.startswith("# -- End  "):
            function_name = line[10:-1]
            if in_function != function_name:
                warn("End %s does not match begin %s" % (function_name, in_function))
            else:
                text += line
                functions.append( (in_function, text) )
            in_function = None
        elif in_function != None:
            text += line
    return functions

def replace_function(file, function, replacement, dest):
    out = open(dest, "w")
    skip = False
    found = False
    in_function = None
    for line in open(file):
        if line.startswith("# -- Begin  "):
            if in_function != None:
                warn("Missing end of function %s" % (in_function,))
            funcname = line[12:-1]
            in_function = funcname
            if in_function == function:
                out.write(replacement)
                skip = True
        elif line.startswith("# -- End  "):
            function_name = line[10:-1]
            if in_function != function_name:
                warn("End %s does not match begin %s" % (function_name, in_function))
            in_function = None
            if skip:
                skip = False
                continue
        if not skip:
            out.write(line)

def announce_test(name):
    stderr.write("%s%s%s: " % (BOLD, name, NORMAL))
    stderr.flush()

def announce_result(result, info):
    stderr.write(result)
    if info != "":
        stderr.write(": %s" % info)
    stderr.write("\n")
    stderr.flush()

def testrun(files):
    linkline="%s %s" % (LINKTEST, " ".join(files),)
    res = subprocess.call(linkline, shell=True)
    if res != 0:
        announce_result(FAILED, "'%s' exitcode != 0" % LINKTEST)
        return False
    else:
        announce_result("ok", "")
        return True

def check_files():
    """Check files mode"""
    for i in range(0, len(NO_PREFIX)):
        f = NO_PREFIX[i]
        b=baddir+"/"+f
        if b not in BAD_FILES:
            warn("There is no corresponding file to '%s' in %s" \
                 % (gooddir+"/"+f, baddir))
            continue

        announce_test(f + " [%s/%s]" % (i+1, len(NO_PREFIX)))

        # combine files (everything from good except f)
        testfiles=[]
        skip=False
        for c in NO_PREFIX:
            badfile = baddir+"/"+c
            goodfile = gooddir+"/"+c
            if c == f:
                testfiles.append(badfile)
                if filecmp.cmp(goodfile, badfile):
                    announce_result("skipped", "same content")
                    skip = True
                    break
            else:
                testfiles.append(goodfile)
        if skip:
            continue
        testrun(testfiles)

def check_functions_in_file(base, goodfile, badfile):
    functions = extract_functions(goodfile)
    if len(functions) == 0:
        warn("Couldn't find any function in %s, missing annotations?" % (goodfile,))
        return
    badfunctions = dict(extract_functions(badfile))
    if len(functions) == 0:
        warn("Couldn't find any function in %s, missing annotations?" % (badfile,))
        return

    COMBINED="/tmp/combined.s"
    i = 0
    for (func,func_text) in functions:
        announce_test(func + " [%s/%s]" % (i+1, len(functions)))
        i+=1
        if func not in badfunctions:
            warn("Function '%s' missing from bad file" % func)
            continue
        if badfunctions[func] == func_text:
            announce_result("skipped", "same content")
            continue
        replace_function(goodfile, func, badfunctions[func], COMBINED)
        testfiles=[]
        for c in NO_PREFIX:
            if c == base:
                testfiles.append(COMBINED)
                continue
            testfiles.append(gooddir + "/" + c)

        testrun(testfiles)

parser = argparse.ArgumentParser()
parser.add_argument('--a', dest='dir_a', default='before')
parser.add_argument('--b', dest='dir_b', default='after')
parser.add_argument('--insane', help='Skip sanity check', action='store_true')
parser.add_argument('file', metavar='file', nargs='?')
config = parser.parse_args()

gooddir=config.dir_a
baddir=config.dir_b

BAD_FILES=find(baddir, "*")
GOOD_FILES=find(gooddir, "*")
NO_PREFIX=sorted([x[len(gooddir)+1:] for x in GOOD_FILES])

# "Checking whether build environment is sane ..."
if not config.insane:
    announce_test("sanity check")
    if not os.access(LINKTEST, os.X_OK):
        error("Expect '%s' to be present and executable" % (LINKTEST,))
        exit(1)

    res = testrun(GOOD_FILES)
    if not res:
        # "build environment is grinning and holding a spatula. Guess not."
        linkline="%s %s" % (LINKTEST, " ".join(GOOD_FILES),)
        stderr.write("\n%s\n\n" % linkline)
        stderr.write("Returned with exitcode != 0\n")
        sys.exit(1)

if config.file is not None:
    # File exchange mode
    goodfile = gooddir+"/"+config.file
    badfile = baddir+"/"+config.file
    check_functions_in_file(config.file, goodfile, badfile)
else:
    # Function exchange mode
    check_files()
