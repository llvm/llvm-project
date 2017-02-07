# RUN: not llvm-mc -arch=hexagon -filetype=obj %s 2>%t; FileCheck %s <%t

# Don't allow a symbolic operand for an insn that cannot take a
# relocation.

r7:6 = rol(r5:4,#r2)

# This should produce an error
#CHECK: error:

