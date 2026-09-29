// RUN: llvm-mc -triple=thumbv7 -show-encoding < %s 2>/dev/null | FileCheck --check-prefix=CHECK-NONARM %s
// RUN: llvm-mc -triple=thumbv8 -show-encoding < %s 2>/dev/null | FileCheck --check-prefix=CHECK-NONARM %s
// RUN: llvm-mc -triple=armv7 -show-encoding < %s 2>/dev/null | FileCheck --check-prefix=CHECK-ARM %s

        // lsl #0 is actually mov, so here we check that it behaves the same as
        // mov with regards to the permitted registers and how it behaves in an
        // IT block.

        // Non-flags-setting with only one of source and destination SP should
        // be OK
        lsl sp, r0, #0
        lsl r0, sp, #0

// CHECK-NONARM: mov.w sp, r0           @ encoding: [0x4f,0xea,0x00,0x0d]
// CHECK-NONARM: mov.w r0, sp           @ encoding: [0x4f,0xea,0x0d,0x00]

// CHECK-ARM: mov sp, r0                @ encoding: [0x00,0xd0,0xa0,0xe1]
// CHECK-ARM: mov r0, sp                @ encoding: [0x0d,0x00,0xa0,0xe1]

        //FIXME: pre-ARMv8 we give an error for these instructions
        //mov sp, r0, lsl #0
        //mov r0, sp, lsl #0

        // LSL #0 in IT block should select the 32-bit encoding
        itt eq
        lsleq  r0, r1, #0
        lslseq r0, r1, #0
        itt gt
        lslgt  r0, r1, #0
        lslsgt r0, r1, #0

// CHECK-NONARM: moveq.w r0, r1         @ encoding: [0x4f,0xea,0x01,0x00]
// CHECK-NONARM: movseq.w r0, r1        @ encoding: [0x5f,0xea,0x01,0x00]
// CHECK-NONARM: movgt.w r0, r1         @ encoding: [0x4f,0xea,0x01,0x00]
// CHECK-NONARM: movsgt.w r0, r1        @ encoding: [0x5f,0xea,0x01,0x00]

// CHECK-ARM: moveq r0, r1              @ encoding: [0x01,0x00,0xa0,0x01]
// CHECK-ARM: movseq r0, r1             @ encoding: [0x01,0x00,0xb0,0x01]
// CHECK-ARM: movgt r0, r1              @ encoding: [0x01,0x00,0xa0,0xc1]
// CHECK-ARM: movsgt r0, r1             @ encoding: [0x01,0x00,0xb0,0xc1]

        itt eq
        moveq  r0, r1, lsl #0
        movseq r0, r1, lsl #0
        itt gt
        movgt  r0, r1, lsl #0
        movsgt r0, r1, lsl #0

// CHECK-NONARM: moveq.w r0, r1         @ encoding: [0x4f,0xea,0x01,0x00]
// CHECK-NONARM: movseq.w r0, r1        @ encoding: [0x5f,0xea,0x01,0x00]
// CHECK-NONARM: movgt.w r0, r1         @ encoding: [0x4f,0xea,0x01,0x00]
// CHECK-NONARM: movsgt.w r0, r1        @ encoding: [0x5f,0xea,0x01,0x00]

// CHECK-ARM: moveq r0, r1              @ encoding: [0x01,0x00,0xa0,0x01]
// CHECK-ARM: movseq r0, r1             @ encoding: [0x01,0x00,0xb0,0x01]
// CHECK-ARM: movgt r0, r1              @ encoding: [0x01,0x00,0xa0,0xc1]
// CHECK-ARM: movsgt r0, r1             @ encoding: [0x01,0x00,0xb0,0xc1]
