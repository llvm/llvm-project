// RUN: not llvm-mc -triple=thumbv8.1m.main -mattr=+mve.fp -mattr=+cdecp0 -mattr=+cdecp1 -show-encoding < %s 2>%t | FileCheck %s
// RUN: FileCheck <%t --check-prefix=ERROR %s

// CHECK: vptete.i8 eq, q0, q0 @ encoding: [0x41,0xfe,0x00,0xef]
vptete.i8 eq, q0, q0
// CHECK-NEXT: vcx1t p0, q1, #1234 @ encoding: [0x29,0xec,0xd2,0x20]
vcx1t    p0, q1, #1234
// CHECK-NEXT: vcx1ae p1, q5, #4095 @ encoding: [0x2f,0xfd,0xff,0xa1]
vcx1ae   p1, q5, #4095
// CHECK-NEXT: vcx2t p1, q0, q6, #123 @ encoding: [0x3e,0xed,0xdc,0x01]
vcx2t    p1, q0, q6, #123
// CHECK-NEXT: vcx2ae p1, q3, q7, #127 @ encoding: [0x3f,0xfd,0xde,0x61]
vcx2ae   p1, q3, q7, #127
// CHECK-NEXT: vpte.i8 eq, q0, q0 @ encoding: [0x41,0xfe,0x00,0x8f]
vpte.i8 eq, q0, q0
// CHECK-NEXT: vcx3at p1, q3, q7, q6, #15 @ encoding: [0xbe,0xfd,0x5c,0x61]
vcx3at   p1, q3, q7, q6, #15
// CHECK-NEXT: vcx3e p0, q0, q2, q0, #12 @ encoding: [0xa4,0xed,0x40,0x00]
vcx3e    p0, q0, q2, q0, #12

vpt.i8 eq, q0, q0
// ERROR: error: incorrect predication in VPT block; got 'none', but expected 't'
vcx1    p0, q1, #1234

vpt.i8 eq, q0, q0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx3t    p0, d0, d1, d7, #1
