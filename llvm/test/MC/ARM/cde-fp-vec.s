// RUN: not llvm-mc -triple=thumbv8m.main -mattr=+fp-armv8 -mattr=+cdecp0 -mattr=+cdecp1 -show-encoding < %s 2>%t | FileCheck %s
// RUN: FileCheck <%t --check-prefixes=ERROR,ERROR-FP %s
// RUN: not llvm-mc -triple=thumbv8m.main -mattr=+fp-armv8d16sp -mattr=+cdecp0 -mattr=+cdecp1 -show-encoding < %s 2>%t | FileCheck %s
// RUN: FileCheck <%t --check-prefixes=ERROR,ERROR-FP %s
// RUN: not llvm-mc -triple=thumbv8.1m.main -mattr=+mve -mattr=+cdecp0 -mattr=+cdecp1 -show-encoding < %s 2>%t | FileCheck --check-prefixes=CHECK,CHECK-MVE %s
// RUN: FileCheck <%t --check-prefixes=ERROR,ERROR-MVE %s

// CHECK-LABEL: test_predication:
test_predication:
ittt eq
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: instructions in IT block must be predicable
vcx1a   p1, s7, #2047
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: instructions in IT block must be predicable
vcx2    p0, d0, d15, #0
// ERROR-FP: [[@LINE+2]]:{{[0-9]+}}: error: instruction requires: mve
// ERROR-MVE: [[@LINE+1]]:{{[0-9]+}}: error: instructions in IT block must be predicable
vcx3    p0, q0, q7, q0, #12
nop
nop
nop

// CHECK-LABEL: test_vcx1:
test_vcx1:
// CHECK-NEXT: vcx1 p0, s11, #1234 @ encoding: [0x69,0xec,0x92,0x50]
vcx1    p0, s11, #1234
// CHECK-NEXT: vcx1a p1, s7, #2047 @ encoding: [0x6f,0xfc,0xbf,0x31]
vcx1a   p1, s7, #2047
// CHECK-NEXT: vcx1 p0, d0, #0 @ encoding: [0x20,0xed,0x00,0x00]
vcx1    p0, d0, #0
// CHECK-NEXT: vcx1a p1, d3, #2047 @ encoding: [0x2f,0xfd,0xbf,0x31]
vcx1a   p1, d3, #2047
// CHECK-MVE-NEXT: vcx1 p0, q1, #1234 @ encoding: [0x29,0xec,0xd2,0x20]
// ERROR-FP: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction, any one of the following would fix this:
vcx1    p0, q1, #1234
// CHECK-MVE-NEXT: vcx1a p1, q5, #4095 @ encoding: [0x2f,0xfd,0xff,0xa1]
// ERROR-FP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
vcx1a   p1, q5, #4095

// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1a   p1, s7, s7, #2047
// ERROR-FP: [[@LINE+4]]:{{[0-9]+}}: error: operand must be an immediate in the range [0,2047]
// ERROR-MVE: [[@LINE+3]]:{{[0-9]+}}: error: invalid instruction, any one of the following would fix this
// ERROR-MVE: [[@LINE+2]]:{{[0-9]+}}: note: operand must be a register in range [q0, q7]
// ERROR-MVE: [[@LINE+1]]:{{[0-9]+}}: note: operand must be an immediate in the range [0,2047]
vcx1    p0, d0, #2048
// ERROR-FP: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an immediate in the range [0,2047]
vcx1a   p1, s0, #2048
// ERROR-FP: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1    p0, q0, #4096
// ERROR-FP: [[@LINE+2]]:{{[0-9]+}}: error: coprocessor must be in the range [p0, p7]
// ERROR-MVE: [[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
vcx1    p8, d0, #1234
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1    p0, d16, #1234
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1    p0, s32, #1234
// ERROR-FP: [[@LINE+7]]:{{[0-9]+}}: error: invalid instruction, any one of the following would fix this:
// ERROR-FP: [[@LINE+6]]:{{[0-9]+}}: note: operand must be a register in range [s0, s31]
// ERROR-FP: [[@LINE+5]]:{{[0-9]+}}: note: operand must be a register in range [d0, d15]
// ERROR-MVE: [[@LINE+4]]:{{[0-9]+}}: error: invalid instruction, any one of the following would fix this:
// ERROR-MVE: [[@LINE+3]]:{{[0-9]+}}: note: operand must be a register in range [q0, q7]
// ERROR-MVE: [[@LINE+2]]:{{[0-9]+}}: note: operand must be a register in range [s0, s31]
// ERROR-MVE: [[@LINE+1]]:{{[0-9]+}}: note: operand must be a register in range [d0, d15]
vcx1    p0, q8, #1234
// ERROR: [[@LINE+3]]:{{[0-9]+}}: error: invalid instruction, any one of the following would fix this:
// ERROR: [[@LINE+2]]:{{[0-9]+}}: note: operand must be a register in range [s0, s31]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: note: operand must be a register in range [d0, d15]
vcx1    p0, r0, #1234
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1    p0, d0, d0, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1a   p0, d0, d2, #0

// CHECK-LABEL: test_vcx2:
test_vcx2:
// CHECK-NEXT: vcx2 p0, s0, s31, #12 @ encoding: [0x33,0xec,0x2f,0x00]
vcx2    p0, s0, s31,  #12
// CHECK-NEXT: vcx2a p0, s1, s1, #63 @ encoding: [0x7f,0xfc,0xb0,0x00]
vcx2a   p0, s1, s1, #63
// CHECK-NEXT: vcx2 p0, d0, d15, #0 @ encoding: [0x30,0xed,0x0f,0x00]
vcx2    p0, d0, d15, #0
// CHECK-NEXT: vcx2a p0, d1, d11, #63 @ encoding: [0x3f,0xfd,0x9b,0x10]
vcx2a   p0, d1, d11, #63
// CHECK-MVE: vcx2 p1, q0, q6, #123 @ encoding: [0x3e,0xed,0xdc,0x01]
vcx2    p1, q0, q6, #123
// CHECK-MVE: vcx2a p1, q3, q7, #127 @ encoding: [0x3f,0xfd,0xde,0x61]
vcx2a   p1, q3, q7, #127

// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an immediate in the range [0,63]
vcx2    p0, d0, d1, #64
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an immediate in the range [0,63]
vcx2a   p0, s3, s1, #64
// ERROR-MVE: [[@LINE+2]]:{{[0-9]+}}: error: operand must be an immediate in the range [0,127]
// ERROR-FP: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx2a   p0, q1, q5, #128
// ERROR-FP: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [d0, d15]
vcx2    p1, d0, q2, #0
// ERROR-FP: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [s0, s31]
vcx2a   p1, q2, s3, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx2    p1, d0, d0, d2, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx2a   p1, q2, q3, q1, #0

// CHECK-LABEL: test_vcx3:
test_vcx3:
// CHECK-NEXT: vcx3 p0, s0, s31, s0, #1 @ encoding: [0x8f,0xec,0x90,0x00]
vcx3    p0, s0, s31, s0, #1
// CHECK-NEXT: vcx3a p1, s1, s17, s11, #7 @ encoding: [0xf8,0xfc,0xb5,0x01]
vcx3a   p1, s1, s17, s11, #7
// CHECK-NEXT: vcx3 p0, d0, d15, d7, #0 @ encoding: [0x8f,0xed,0x07,0x00]
vcx3    p0, d0, d15, d7, #0
// CHECK-NEXT: vcx3a p1, d1, d11, d11, #7 @ encoding: [0xbb,0xfd,0x1b,0x11]
vcx3a   p1, d1, d11, d11, #7
// CHECK-MVE-NEXT: vcx3 p0, q0, q2, q0, #12 @ encoding: [0xa4,0xed,0x40,0x00]
vcx3    p0, q0, q2, q0, #12
// CHECK-MVE-NEXT: vcx3a p1, q3, q7, q6, #15 @ encoding: [0xbe,0xfd,0x5c,0x61]
vcx3a   p1, q3, q7, q6, #15

// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an immediate in the range [0,7]
vcx3a   p1, d1, d11, d12, #8
// ERROR-MVE: [[@LINE+2]]:{{[0-9]+}}: error: operand must be an immediate in the range [0,15]
// ERROR-FP: error: invalid instruction
vcx3a   p1, q1, q2, q3, #16
// ERROR-MVE: [[@LINE+2]]:{{[0-9]+}}: error: operand must be a register in range [d0, d15]
// ERROR-FP: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [d0, d15]
vcx3    p0, d0, q0, d7, #1
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [s0, s31]
vcx3a   p1, s0, s1, d3, #2
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx3a   p0, s0, d0, q0, #2
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx3    p0, s0, s0, s31, s0, #1
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx3a   p1, d1, d3, d22, d22, #7
