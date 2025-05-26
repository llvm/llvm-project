// RUN: not llvm-mc -triple=x86_64 %s 2> %t
// RUN: FileCheck --input-file %t %s

        .data

        t1_v1 = 1
        t1_v1 = 2

t2_s0:
// CHECK: redefinition of 't2_s0'
        t2_s0 = 2

        t3_s0 = t2_s0 + 1
        .long t3_s0
// CHECK: invalid reassignment of non-absolute variable 't3_s0'
        t3_s0 = 1

