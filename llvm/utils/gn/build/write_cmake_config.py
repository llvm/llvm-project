#!/usr/bin/env python
"""Emulates the bits of CMake's configure_file() function needed in LLVM.

The CMake build uses configure_file() for several things.  This emulates that
function for the GN build.  In the GN build, this runs at build time, instead
of at generator time.

Takes a list of KEY=VALUE pairs (where VALUE can be empty).

On each line, replaces ${KEY} with VALUE.

After that, also handles these special cases (note that FOO= sets the value of
FOO to the empty string, which is falsy, but FOO=0 sets it to '0' which is
truthy):

1.) #cmakedefine01 FOO
    Checks if key FOO is set to a truthy value, and depending on that prints
    one of the following two lines:

        #define FOO 1
        #define FOO 0

2.) #cmakedefine FOO [...]
    Checks if key FOO is set to a truthy in value, and depending on that prints
    one of the following two lines:

        #define FOO [...]
        /* #undef FOO */

Fails if any of the KEY=VALUE arguments aren't needed for processing the
.h.cmake file, or if the .h.cmake file has unreplaced ${VAR} references after
processing all values.
"""

import argparse
import os
import re
import sys


def main():
    parser = argparse.ArgumentParser(
                 epilog=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='input file')
    parser.add_argument('values', nargs='*', help='several KEY=VALUE pairs')
    parser.add_argument('-o', '--output', required=True,
                        help='output file')
    args = parser.parse_args()

    values = {}
    for value in args.values:
        key, val = value.split('=', 1)
        values[key] = val
    unused_values = set(values.keys())

    # Matches e.g. '${CLANG_FOO}' and captures CLANG_FOO in group 1.
    var_re = re.compile(r'\$\{([^}]*)\}')

    in_lines = open(args.input).readlines()
    out_lines = []
    for in_line in in_lines:
        def repl(m):
            unused_values.discard(m.group(1))
            return values[m.group(1)]
        in_line = var_re.sub(repl, in_line)
        if in_line.startswith('#cmakedefine01 '):
            _, var = in_line.split()
            in_line = '#define %s %d\n' % (var, 1 if values[var] else 0)
            unused_values.discard(var)
        elif in_line.startswith('#cmakedefine '):
            _, var = in_line.split(None, 1)
            try:
                var, val = var.split(None, 1)
                in_line = '#define %s %s' % (var, val)  # val ends in \n.
            except:
                var = var.rstrip()
                in_line = '#define %s\n' % var
            if not values[var]:
                in_line = '/* #undef %s */\n' % var
            unused_values.discard(var)
        out_lines.append(in_line)

    if unused_values:
        print >>sys.stderr, 'Unused --values args:'
        print >>sys.stderr, '    ', '\n    '.join(unused_values)
        return 1

    output = ''.join(out_lines)

    leftovers = var_re.findall(output)
    if leftovers:
        print >>sys.stderr, 'unprocessed values:\n', '\n'.join(leftovers)
        return 1

    if not os.path.exists(args.output) or open(args.output).read() != output:
        open(args.output, 'w').write(output)


if __name__ == '__main__':
    sys.exit(main())
