# RUN: llvm-mc -triple s390x-linux-gnu -mcpu=z14 -filetype=obj %s | \
# RUN: llvm-objdump --mcpu=z14 -d - | FileCheck %s

# Test the .insn directive which provides a way of encoding an instruction
# directly. It takes a format, encoding, and operands based on the format.
# This file covers instruction formats newly supported.

label.START:
#CHECK: e6 12 30 45 60 71      vap %v1, %v2, %v3, 86, 4
      .insn vri_f,0xe60000000071,%v1,%v2,%v3,86,4


#CHECK: e6 12 34 56 70 5b      vpsop %v1, %v2, 103, 52, 5
      .insn vri_g,0xe6000000005b,%v1,%v2,103,52,5

#CHECK: e6 10 34 56 70 49      vlip %v1, 13398, 7
      .insn vri_h,0xe60000000049,%v1,13398,7,0

#CHECK: e6 12 00 34 50 58      vcvd %v1, %r2, 69, 3
      .insn vri_i,0xe60000000058,%v1,%r2,69,3

#CHECK: e6 12 00 30 00 50      vcvb %r1, %v2, 3
      .insn vrr_i,0xe60000000050,%r1,%v2,3,0

#CHECK: e6 01 23 45 60 37      vlrlr %v6, %r1, 837(%r2)
      .insn vrs_d,0xe60000000037,%v6,837(%r2),%r1

#CHECK: e6 0f 00 00 00 5f      vtp %v15
      .insn vrr_g,0xe6000000005f,%v15,0

#CHECK: e6 01 20 30 00 77      vcp %v1, %v2, 3
      .insn vrr_h,0xe60000000077,%v1,%v2,3

