# Special test to make sure we don't error on VFP co-proc access
@ RUN: llvm-mc -triple=armv5 < %s | FileCheck %s
@ RUN: llvm-mc -triple=armv6 < %s | FileCheck %s

        @ p10 and p11 are reserved for NEON, but accessible on v5/v6
        ldc  p10, cr0, [r0], {0x20}
        ldc2 p11, cr0, [r0], {0x21}
        ldcl p11, cr0, [r0], {0x20}

@ CHECK-NOT: error: invalid operand for instruction
