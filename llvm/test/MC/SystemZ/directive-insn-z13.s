# RUN: llvm-mc -triple s390x-linux-gnu -mcpu=z13 -filetype=obj %s | \
# RUN: llvm-objdump --mcpu=z13 -d - | FileCheck %s

# Test the .insn directive which provides a way of encoding an instruction
# directly. It takes a format, encoding, and operands based on the format.
# This file covers instruction formats newly supported.

label.START:
#CHECK: ec 12 af fe 00 42    	lochih	%r1, -20482
      .insn rie_g,0xec0000000042,%r1,-20482,2

#CHECK: ed 12 34 56 78 ae    	cdpt	%f7, 1110(19,%r3), 8
      .insn rsl_b,0xed00000000ae,%r7,1110(19,%r3),8

#CHECK: eb 12 34 56 78 23    	clth	%r1, 492630(%r3)
      .insn rsy_b,0xeb0000000023,%r1,492630(%r3),2

#CHECK: c7 00 12 34 56 78     bpp 0, 0xad02, 564(%r1)
      .insn smi,0xc70000000000,0,44272,564(%r1)

#CHECK: e7 10 23 45 00 44      vgbm %v1, 9029
      .insn vri_a,0xe70000000044,%v1,9029,0

#CHECK: e7 10 ff fc 00 44      vgbm %v1, 65532
      .insn vri_a,0xe70000000044,%v1,-4,0

#CHECK: e7 10 23 45 60 46      vgm %v1, 35, 69, 6
      .insn vri_b,0xe70000000046,%v1,35,69,6

#CHECK: e7 12 34 56 70 4d      vrep %v1, %v2, 13398, 7
      .insn vri_c,0xe7000000004d,%v1,13398,%v2,7

#CHECK: e7 12 30 45 60 72      verim %v1, %v2, %v3, 69, 6
      .insn vri_d,0xe70000000072,%v1,%v2,%v3,69,6

#CHECK: e7 12 34 56 70 4a      vftci %v1, %v2, 837, 7, 6
      .insn vri_e,0xe7000000004a,%v1,%v2,837,7,6

#CHECK: e7 12 30 40 50 97      vpks %v1, %v2, %v3, 5, 4
      .insn vrr_b,0xe70000000097,%v1,%v2,%v3,5,4

#CHECK: e7 12 30 00 40 94      vpk %v1, %v2, %v3, 4
      .insn vrr_c,0xe70000000094,%v1,%v2,%v3,4,0,0

#CHECK: e7 12 34 00 50 bb      vacq %v1, %v2, %v3, %v5
      .insn vrr_d,0xe700000000bb,%v1,%v2,%v3,%v5,4,0

#CHECK: e7 12 34 05 60 8f      vfma %v1, %v2, %v3, %v6, 5, 4
      .insn vrr_e,0xe7000000008f,%v1,%v2,%v3,%v6,5,4

#CHECK: e7 12 34 56 78 36      vlm %v17, %v2, 1110(%r3), 7
      .insn vrs_a,0xe70000000036,%v17,1110(%r3),%v2,7

#CHECK: e7 12 34 56 78 22      vlvg %v17, %r2, 1110(%r3), 7
      .insn vrs_b,0xe70000000022,%v17,1110(%r3),%r2,7

#CHECK: e7 12 34 56 70 21      vlgv %r1, %v2, 1110(%r3), 7
      .insn vrs_c,0xe70000000021,%r1,1110(%r3),%v2,7

