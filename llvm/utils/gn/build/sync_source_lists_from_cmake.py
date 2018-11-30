#!/usr/bin/env python

"""Helps to keep BUILD.gn files in sync with the corresponding CMakeLists.txt.

For each BUILD.gn file in the tree, checks if the list of cpp files in
it is identical to the list of cpp files in the corresponding CMakeLists.txt
file, and prints the difference if not."""

from __future__ import print_function

import os
import re
import subprocess

def main():
    gn_files = subprocess.check_output(
            ['git', 'ls-files', '*BUILD.gn']).splitlines()

    # Matches e.g. |   "foo.cpp",|.
    gn_cpp_re = re.compile(r'^\s*"([^"]+\.(?:cpp|h))",$', re.MULTILINE)
    # Matches e.g. |   "foo.cpp"|.
    cmake_cpp_re = re.compile(r'^\s*([A-Za-z_0-9/-]+\.(?:cpp|h))$',
                              re.MULTILINE)

    for gn_file in gn_files:
        # The CMakeLists.txt for llvm/utils/gn/secondary/foo/BUILD.gn is
        # directly at foo/CMakeLists.txt.
        strip_prefix = 'llvm/utils/gn/secondary/'
        if not gn_file.startswith(strip_prefix):
            continue
        cmake_file = os.path.join(
                os.path.dirname(gn_file[len(strip_prefix):]), 'CMakeLists.txt')
        if not os.path.exists(cmake_file):
            continue

        def get_sources(source_re, text):
            return set([m.group(1) for m in source_re.finditer(text)])
        gn_cpp = get_sources(gn_cpp_re, open(gn_file).read())
        cmake_cpp = get_sources(cmake_cpp_re, open(cmake_file).read())

        if gn_cpp == cmake_cpp:
            continue

        print(gn_file)
        add = cmake_cpp - gn_cpp
        if add:
            print('add:\n' + '\n'.join('    "%s",' % a for a in add))
        remove = gn_cpp - cmake_cpp
        if remove:
            print('remove:\n' + '\n'.join(remove))
        print()

if __name__ == '__main__':
    main()
