#!/usr/bin/env python3
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

import os
import sys

if (len(sys.argv) != 4):
    print('usage: ' + sys.argv[0] + ' <LLVM major version> <input> <output>')
    sys.exit(1)

with open(sys.argv[3], 'w') as out_fd:
    out_fd.write('LLVM_' + sys.argv[1] + ' {\n')
    if os.stat(sys.argv[2]).st_size > 0:
        out_fd.write('  global:\n')
        with open(sys.argv[2], 'r') as in_fd:
            for e in in_fd.readlines():
                out_fd.write('    ' + e.rstrip() + ';\n')
    out_fd.write('  local: *;\n};\n')

sys.exit(0)
