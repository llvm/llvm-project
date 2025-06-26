# RUN: llvm-mc -triple s390x-linux-gnu -mcpu=z17 -filetype=obj %s | \
# RUN: llvm-objdump --mcpu=z17 -d - | FileCheck %s

# Test the .insn directive which provides a way of encoding an instruction
# directly. It takes a format, encoding, and operands based on the format.
# This file covers instruction formats newly supported.

label.START:
#CHECK: e3 12 34 56 78 60     lxab %r1, 492630(%r2,%r3)
      .insn rxy_c,0xe30000000060,%r1,492630(%r2,%r3)

#CHECK: e6 12 00 34 50 4a      vcvdq %v1, %v2, 69, 3
      .insn vri_j,0xe6000000004a,%v1,%v2,69,3

#CHECK: e7 12 30 45 60 88      veval %v1, %v2, %v3, %v6, 69
      .insn vri_k,0xe70000000088,%v1,%v2,%v3,%v6,69

#CHECK: e6 01 23 45 60 7f      vtz %v1, %v2, 13398
      .insn vri_l,0xe6000000007f,%v1,%v2,13398

#CHECK: e7 12 00 00 30 54      vgemg %v1, %v2
      .insn vrr_a,0xe70000000054,%v1,%v2,3,0,0

