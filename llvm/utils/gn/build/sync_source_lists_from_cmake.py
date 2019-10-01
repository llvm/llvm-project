#!/usr/bin/env python

"""Helps to keep BUILD.gn files in sync with the corresponding CMakeLists.txt.

For each BUILD.gn file in the tree, checks if the list of cpp files in
it is identical to the list of cpp files in the corresponding CMakeLists.txt
file, and prints the difference if not.

Also checks that each CMakeLists.txt file below unittests/ folders that define
binaries have corresponding BUILD.gn files.
"""

from __future__ import print_function

import os
import re
import subprocess


def sync_source_lists():
    # Use shell=True on Windows in case git is a bat file.
    gn_files = subprocess.check_output(['git', 'ls-files', '*BUILD.gn'],
                                       shell=os.name == 'nt').splitlines()

    # Matches e.g. |   "foo.cpp",|, captures |foo| in group 1.
    gn_cpp_re = re.compile(r'^\s*"([^"]+\.(?:cpp|c|h|S))",$', re.MULTILINE)
    # Matches e.g. |   foo.cpp|, captures |foo| in group 1.
    cmake_cpp_re = re.compile(r'^\s*([A-Za-z_0-9./-]+\.(?:cpp|c|h|S))$',
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


def sync_unittests():
    # Matches e.g. |add_llvm_unittest_with_input_files|.
    unittest_re = re.compile(r'^add_\S+_unittest', re.MULTILINE)

    checked = [ 'clang', 'clang-tools-extra', 'lld', 'llvm' ]
    for c in checked:
        for root, _, _ in os.walk(os.path.join(c, 'unittests')):
            cmake_file = os.path.join(root, 'CMakeLists.txt')
            if not os.path.exists(cmake_file):
                continue
            if not unittest_re.search(open(cmake_file).read()):
                continue  # Skip CMake files that just add subdirectories.
            gn_file = os.path.join('llvm/utils/gn/secondary', root, 'BUILD.gn')
            if not os.path.exists(gn_file):
                print('missing GN file %s for unittest CMake file %s' %
                      (gn_file, cmake_file))


def main():
    sync_source_lists()
    sync_unittests()


if __name__ == '__main__':
    main()
