# RUN: llvm-mc -triple s390x-linux-gnu -mcpu=z16 -filetype=obj %s | \
# RUN: llvm-objdump --mcpu=z16 -d - | FileCheck %s

# Test the .insn directive which provides a way of encoding an instruction
# directly. It takes a format, encoding, and operands based on the format.
# This file covers instruction formats newly supported.

label.START:
#CHECK: e6 12 30 40 00 7d      vcsph %v1, %v2, %v3, 4
      .insn vrr_j,0xe6000000007d,%v1,%v2,%v3,4

#CHECK: e6 12 00 30 00 51      vclzdp %v1, %v2, 3
      .insn vrr_k,0xe60000000051,%v1,%v2,3

