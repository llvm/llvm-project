// RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t
// RUN: FileCheck --input-file %t %s

        .data
// CHECK: Recursive use of 't0_v0'
        t0_v0 = t0_v0 + 1

        t1_v1 = 1
        t1_v1 = 2

t2_s0:
// CHECK: redefinition of 't2_s0'
        t2_s0 = 2

        t3_s0 = t2_s0 + 1
        .long t3_s0
// CHECK: invalid reassignment of non-absolute variable 't3_s0'
        t3_s0 = 1


// CHECK: Recursive use of 't4_s2'
        t4_s0 = t4_s1
        t4_s1 = t4_s2
        t4_s2 = t4_s0

// CHECK: Recursive use of 't5_s1'
        t5_s0 = t5_s1 + 1
        t5_s1 = t5_s0
