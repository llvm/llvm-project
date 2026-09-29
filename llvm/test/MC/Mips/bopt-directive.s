# RUN: llvm-mc -triple=mips -mcpu=mips32 %s 2>&1 | FileCheck %s

# We don't support the bopt option in the integrated assembler. Given it's
# single pass nature, it would be quite difficult to implement currently.

# Ensure we parse the bopt & nobopt directives and warn in the bopt case.

# CHECK: warning: 'bopt' feature is unsupported
# CHECK: nop
.text
f:
.set bopt
g:
.set nobopt
nop

