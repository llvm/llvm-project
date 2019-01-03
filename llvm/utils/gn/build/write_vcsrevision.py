#!/usr/bin/env python

"""Gets the current revision and writes it to VCSRevision.h."""

from __future__ import print_function

import argparse
import os
import subprocess
import sys


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
LLVM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))
MONO_DIR = os.path.dirname(LLVM_DIR)


def which(program):
    # distutils.spawn.which() doesn't find .bat files,
    # https://bugs.python.org/issue2200
    for path in os.environ["PATH"].split(os.pathsep):
        candidate = os.path.join(path, program)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--depfile',
                        help='if set, writes a depfile that causes this script '
                             'to re-run each time the current revision changes')
    parser.add_argument('vcs_header', help='path to the output file to write')
    args = parser.parse_args()

    if os.path.isdir(os.path.join(LLVM_DIR, '.svn')):
        print('SVN support not implemented', file=sys.stderr)
        return 1
    if os.path.isdir(os.path.join(LLVM_DIR, '.git')):
        print('non-mono-repo git support not implemented', file=sys.stderr)
        return 1

    git_dir = os.path.join(MONO_DIR, '.git')
    if not os.path.isdir(git_dir):
        print('.git dir not found at "%s"' % git_dir, file=sys.stderr)
        return 1

    git, use_shell = which('git'), False
    if not git:
        git = which('git.exe')
    if not git:
        git = which('git.bat')
        use_shell = True
    rev = subprocess.check_output([git, 'rev-parse', '--short', 'HEAD'],
                                  cwd=git_dir, shell=use_shell)
    # FIXME: add pizzas such as the svn revision read off a git note?
    vcsrevision_contents = '#define LLVM_REVISION "git-%s"\n' % rev.strip()

    # If the output already exists and is identical to what we'd write,
    # return to not perturb the existing file's timestamp.
    if os.path.exists(args.vcs_header) and \
            open(args.vcs_header).read() == vcsrevision_contents:
        return 0

    # http://neugierig.org/software/blog/2014/11/binary-revisions.html
    if args.depfile:
        build_dir = os.getcwd()
        with open(args.depfile, 'w') as depfile:
            depfile.write('%s: %s\n' % (
                args.vcs_header,
                os.path.relpath(os.path.join(git_dir, 'logs', 'HEAD'),
                                build_dir)))
    open(args.vcs_header, 'w').write(vcsrevision_contents)


if __name__ == '__main__':
    sys.exit(main())
