// RUN: not llvm-mc -triple aarch64 -mattr +brbe -show-encoding %s 2>%t | FileCheck %s
// RUN: FileCheck --check-prefix=ERROR %s < %t
// RUN: not llvm-mc -triple aarch64 -show-encoding %s 2>%t
// RUN: FileCheck --check-prefix=ERROR-NO-BRBE %s < %t

msr BRBCR_EL1, x0
mrs x1, BRBCR_EL1
// CHECK: msr     BRBCR_EL1, x0           // encoding: [0x00,0x90,0x11,0xd5]
// CHECK: mrs     x1, BRBCR_EL1           // encoding: [0x01,0x90,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:9: error: expected readable system register

msr BRBCR_EL12, x2
mrs x3, BRBCR_EL12
// CHECK: msr     BRBCR_EL12, x2          // encoding: [0x02,0x90,0x15,0xd5]
// CHECK: mrs     x3, BRBCR_EL12          // encoding: [0x03,0x90,0x35,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:9: error: expected readable system register

msr BRBCR_EL2, x4
mrs x5, BRBCR_EL2
// CHECK: msr     BRBCR_EL2, x4           // encoding: [0x04,0x90,0x14,0xd5]
// CHECK: mrs     x5, BRBCR_EL2           // encoding: [0x05,0x90,0x34,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:9: error: expected readable system register

msr BRBFCR_EL1, x6
mrs x7, BRBFCR_EL1
// CHECK: msr     BRBFCR_EL1, x6          // encoding: [0x26,0x90,0x11,0xd5]
// CHECK: mrs     x7, BRBFCR_EL1          // encoding: [0x27,0x90,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:9: error: expected readable system register

msr BRBIDR0_EL1, x8
mrs x9, BRBIDR0_EL1
// ERROR: [[@LINE-2]]:5: error: expected writable system register
// CHECK: mrs     x9, BRBIDR0_EL1         // encoding: [0x09,0x92,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:9: error: expected readable system register

msr BRBINFINJ_EL1, x10
mrs x11, BRBINFINJ_EL1
// CHECK: msr     BRBINFINJ_EL1, x10      // encoding: [0x0a,0x91,0x11,0xd5]
// CHECK: mrs     x11, BRBINFINJ_EL1      // encoding: [0x0b,0x91,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:10: error: expected readable system register

msr BRBSRCINJ_EL1, x12
mrs x13, BRBSRCINJ_EL1
// CHECK: msr     BRBSRCINJ_EL1, x12      // encoding: [0x2c,0x91,0x11,0xd5]
// CHECK: mrs     x13, BRBSRCINJ_EL1      // encoding: [0x2d,0x91,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:10: error: expected readable system register

msr BRBTGTINJ_EL1, x14
mrs x15, BRBTGTINJ_EL1
// CHECK: msr     BRBTGTINJ_EL1, x14      // encoding: [0x4e,0x91,0x11,0xd5]
// CHECK: mrs     x15, BRBTGTINJ_EL1      // encoding: [0x4f,0x91,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:10: error: expected readable system register

msr BRBTS_EL1, x16
mrs x17, BRBTS_EL1
// CHECK: msr     BRBTS_EL1, x16          // encoding: [0x50,0x90,0x11,0xd5]
// CHECK: mrs     x17, BRBTS_EL1          // encoding: [0x51,0x90,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:10: error: expected readable system register

// Rather than testing all 32 registers in the three BRBINF/BRBSRC/BRBTGT
// families, I'll test a representative sample, including all bits clear,
// all bits set, each bit set individually, and a couple of intermediate
// patterns.

msr BRBINF0_EL1, x18
mrs x19, BRBINF0_EL1
// ERROR: [[@LINE-2]]:5: error: expected writable system register
// CHECK: mrs     x19, BRBINF0_EL1        // encoding: [0x13,0x80,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:10: error: expected readable system register

msr BRBINF1_EL1, x20
mrs x21, BRBINF1_EL1
// ERROR: [[@LINE-2]]:5: error: expected writable system register
// CHECK: mrs     x21, BRBINF1_EL1        // encoding: [0x15,0x81,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:10: error: expected readable system register

msr BRBINF2_EL1, x22
mrs x23, BRBINF2_EL1
// ERROR: [[@LINE-2]]:5: error: expected writable system register
// CHECK: mrs     x23, BRBINF2_EL1        // encoding: [0x17,0x82,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:10: error: expected readable system register

msr BRBSRC4_EL1, x24
mrs x25, BRBSRC4_EL1
// ERROR: [[@LINE-2]]:5: error: expected writable system register
// CHECK: mrs     x25, BRBSRC4_EL1        // encoding: [0x39,0x84,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:10: error: expected readable system register

msr BRBSRC8_EL1, x26
mrs x27, BRBSRC8_EL1
// ERROR: [[@LINE-2]]:5: error: expected writable system register
// CHECK: mrs     x27, BRBSRC8_EL1        // encoding: [0x3b,0x88,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:10: error: expected readable system register

msr BRBSRC16_EL1, x28
mrs x29, BRBSRC16_EL1
// ERROR: [[@LINE-2]]:5: error: expected writable system register
// CHECK: mrs     x29, BRBSRC16_EL1       // encoding: [0xbd,0x80,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:10: error: expected readable system register

msr BRBTGT10_EL1, x0
mrs x1, BRBTGT10_EL1
// ERROR: [[@LINE-2]]:5: error: expected writable system register
// CHECK: mrs     x1, BRBTGT10_EL1        // encoding: [0x41,0x8a,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:9: error: expected readable system register

msr BRBTGT21_EL1, x2
mrs x3, BRBTGT21_EL1
// ERROR: [[@LINE-2]]:5: error: expected writable system register
// CHECK: mrs     x3, BRBTGT21_EL1        // encoding: [0xc3,0x85,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:9: error: expected readable system register

msr BRBTGT31_EL1, x4
mrs x5, BRBTGT31_EL1
// ERROR: [[@LINE-2]]:5: error: expected writable system register
// CHECK: mrs     x5, BRBTGT31_EL1        // encoding: [0xc5,0x8f,0x31,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:5: error: expected writable system register
// ERROR-NO-BRBE: [[@LINE-4]]:9: error: expected readable system register

brb iall
brb inj
// CHECK: brb iall  // encoding: [0x9f,0x72,0x09,0xd5]
// CHECK: brb inj   // encoding: [0xbf,0x72,0x09,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:1: error: instruction requires: brbe
// ERROR-NO-BRBE: [[@LINE-4]]:1: error: instruction requires: brbe

brb IALL
brb INJ
// CHECK: brb iall  // encoding: [0x9f,0x72,0x09,0xd5]
// CHECK: brb inj   // encoding: [0xbf,0x72,0x09,0xd5]
// ERROR-NO-BRBE: [[@LINE-4]]:1: error: instruction requires: brbe
// ERROR-NO-BRBE: [[@LINE-4]]:1: error: instruction requires: brbe
