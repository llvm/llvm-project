# RUN: llvm-mc --triple=loongarch32 --mattr=+f --loongarch-numeric-reg %s \
# RUN:     | FileCheck %s
# RUN: llvm-mc --triple=loongarch32 --mattr=+f -M numeric %s \
# RUN:     | FileCheck %s
# RUN: llvm-mc --triple=loongarch32 --mattr=+f --filetype=obj %s -o %t.32
# RUN: llvm-objdump -d -M numeric %t.32 | FileCheck %s
# RUN: llvm-mc --triple=loongarch64 --mattr=+f --loongarch-numeric-reg %s \
# RUN:     | FileCheck %s
# RUN: llvm-mc --triple=loongarch64 --mattr=+f -M numeric %s \
# RUN:     | FileCheck %s
# RUN: llvm-mc --triple=loongarch64 --mattr=+f --filetype=obj %s -o %t.64
# RUN: llvm-objdump -d -M numeric %t.64 | FileCheck %s

addi.w $zero, $ra, 1
addi.w $tp, $sp, 1
addi.w $a0, $a1, 1
addi.w $a2, $a3, 1
addi.w $a4, $a5, 1
addi.w $a6, $a7, 1
addi.w $t0, $t1, 1
addi.w $t2, $t3, 1
addi.w $t4, $t5, 1
addi.w $t6, $t7, 1
addi.w $t8, $r21, 1
addi.w $fp, $s0, 1
addi.w $s1, $s2, 1
addi.w $s3, $s4, 1
addi.w $s5, $s6, 1
addi.w $s7, $s8, 1

# CHECK:      addi.w  $r0, $r1, 1
# CHECK-NEXT: addi.w  $r2, $r3, 1
# CHECK-NEXT: addi.w  $r4, $r5, 1
# CHECK-NEXT: addi.w  $r6, $r7, 1
# CHECK-NEXT: addi.w  $r8, $r9, 1
# CHECK-NEXT: addi.w  $r10, $r11, 1
# CHECK-NEXT: addi.w  $r12, $r13, 1
# CHECK-NEXT: addi.w  $r14, $r15, 1
# CHECK-NEXT: addi.w  $r16, $r17, 1
# CHECK-NEXT: addi.w  $r18, $r19, 1
# CHECK-NEXT: addi.w  $r20, $r21, 1
# CHECK-NEXT: addi.w  $r22, $r23, 1
# CHECK-NEXT: addi.w  $r24, $r25, 1
# CHECK-NEXT: addi.w  $r26, $r27, 1
# CHECK-NEXT: addi.w  $r28, $r29, 1
# CHECK-NEXT: addi.w  $r30, $r31, 1

fmadd.s $fa0, $fa1, $fa2, $fa3
fmadd.s $fa4, $fa5, $fa6, $fa7
fmadd.s $ft0, $ft1, $ft2, $ft3
fmadd.s $ft4, $ft5, $ft6, $ft7
fmadd.s $ft8, $ft9, $ft10, $ft11
fmadd.s $ft12, $ft13, $ft14, $ft15
fmadd.s $fs0, $fs1, $fs2, $fs3
fmadd.s $fs4, $fs5, $fs6, $fs7

# CHECK:      fmadd.s $f0, $f1, $f2, $f3
# CHECK-NEXT: fmadd.s $f4, $f5, $f6, $f7
# CHECK-NEXT: fmadd.s $f8, $f9, $f10, $f11
# CHECK-NEXT: fmadd.s $f12, $f13, $f14, $f15
# CHECK-NEXT: fmadd.s $f16, $f17, $f18, $f19
# CHECK-NEXT: fmadd.s $f20, $f21, $f22, $f23
# CHECK-NEXT: fmadd.s $f24, $f25, $f26, $f27
# CHECK-NEXT: fmadd.s $f28, $f29, $f30, $f31
