#!/usr/bin/env python
# Strip unnecessary data from a core dump to reduce its size.
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os.path
import sys
import tempfile

from elftools.elf.elffile import ELFFile


def strip_non_notes(elf, inf, outf):
    next_segment_offset = min(x.header.p_offset for x in elf.iter_segments())
    copy_segments = filter(lambda x: x.header.p_type == "PT_NOTE", elf.iter_segments())

    # first copy the headers
    inf.seek(0)
    outf.write(inf.read(next_segment_offset))

    for seg in copy_segments:
        assert seg.header.p_filesz > 0

        inf.seek(seg.header.p_offset)
        # fill the area between last write and new offset with zeros
        outf.seek(seg.header.p_offset)

        # now copy the segment
        outf.write(inf.read(seg.header.p_filesz))


def main():
    argp = argparse.ArgumentParser()
    action = argp.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--strip-non-notes",
        action="store_const",
        const=strip_non_notes,
        dest="action",
        help="Strip all segments except for notes",
    )
    argp.add_argument("elf", help="ELF file to strip (in place)", nargs="+")
    args = argp.parse_args()

    for path in args.elf:
        with open(path, "rb") as f:
            elf = ELFFile(f)
            # we do not support copying the section table now
            assert elf.num_sections() == 0

            tmpf = tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False)
            try:
                args.action(elf, f, tmpf)
            except:
                os.unlink(tmpf.name)
                raise
            else:
                os.rename(tmpf.name, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
