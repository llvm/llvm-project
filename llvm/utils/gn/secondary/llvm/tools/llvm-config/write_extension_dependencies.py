#!/usr/bin/env python
r"""Writes ExtensionDepencencies.inc."""

from __future__ import print_function

import argparse
import os
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', required=True,
                        help='output file')
    args = parser.parse_args()

    source = """\
#include <array>
struct ExtensionDescriptor {
  const char* Name;
  const char* const RequiredLibraries[1 + 1];
};
std::array<ExtensionDescriptor, 0>  AvailableExtensions{};
"""
    open(args.output, 'w').write(source)


if __name__ == '__main__':
    sys.exit(main())
