#!/usr/bin/env python

import argparse
import re
import sys


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('infile', type=argparse.FileType('rb'),
                      help='Input vmcore file')
    argp.add_argument('outfile', type=argparse.FileType('wb'),
                      help='Output vmcore file')
    args = argp.parse_args()

    inf = args.infile
    outf = args.outfile
    line_re = re.compile(r"^% RD: (\d+) (\d+)")

    # copy the first chunk that usually includes ELF headers
    # (not output by patched libfbsdvmcore since libelf reads this)
    outf.write(inf.read(1024))

    for l in sys.stdin:
        m = line_re.match(l)
        if m is None:
            continue
        offset, size = [int(x) for x in m.groups()]

        inf.seek(offset)
        outf.seek(offset)
        outf.write(inf.read(size))


if __name__ == "__main__":
    main()
