#! /usr/bin/python
#
#  Main authors:
#    Roberto Castaneda Lozano <roberto.castaneda@ri.se>
#
#  This file is part of Unison, see http://unison-code.github.io
#
#  Copyright (c) 2018, RISE SICS AB
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#

# llc-unison: script to run llc with Unison
#
# Runs llc twice to generate Unison's input, then Unison itself, and then llc
# again to emit the generated assembly code. Has the same interface as llc
# itself, plus a few additional flags to control Unison.

import os
import sys
import argparse
import subprocess
import tempfile

def execute(cmd):
    print(" ".join(cmd))
    subprocess.call(cmd)
    return

def temp_filename(ext):
    return tempfile.NamedTemporaryFile(suffix=ext).name

# Intercept input file, output file, and Unison flags
parser = argparse.ArgumentParser(description='Run llc with Unison. The option -o must be given.')
parser.add_argument('infile', metavar='INPUT', help='input file')
parser.add_argument('-o', metavar='OUTPUT', help='output file')
parser.add_argument('--uni-flags', help='flags to be passed to Unison')
(args, llc_flags) = parser.parse_known_args()

exit_pass  = "phi-node-elimination"
entry_pass = "funclet-layout"

# Expect 'llc' in the same directory
llc = os.path.join(os.path.dirname(sys.argv[0]), "llc")
# Expect 'uni' in the PATH
uni = "uni"

# Generate main input to Unison (.ll -> .mir)

mir = temp_filename('.mir')
cmd_mir = [llc] + llc_flags + \
          ["-stop-before", exit_pass, "-unison-mir", "-o", mir, args.infile]
execute(cmd_mir)

# Generate initial solution for Unison (.ll -> .asm.mir)

asm_mir = temp_filename('.asm.mir')
cmd_asm_mir = [llc] + llc_flags + \
              ["-stop-before", entry_pass, "-unison-mir", "-o", asm_mir, args.infile]
execute(cmd_asm_mir)

# Run Unison (.mir -> .asm.mir -> .unison.mir)

unison_mir = temp_filename('.unison.mir')
cmd_uni = [uni, "run", "--llvm6", "--verbose"] + \
          ["-o", unison_mir, mir, "--basefile=" + asm_mir]
if args.uni_flags is not None:
    cmd_uni += [args.uni_flags]
execute(cmd_uni)

# Generate assembly code (.unison.mir -> .s)

cmd_s = [llc] + llc_flags + \
        ["-start-before", entry_pass, "-o", args.o, unison_mir]
execute(cmd_s)