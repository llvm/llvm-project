// RUN:     llvm-mc -triple armv8a   -show-encoding -mattr=+i8mm < %s      | FileCheck %s --check-prefix=ARM
// RUN:     llvm-mc -triple thumbv8a -show-encoding -mattr=+i8mm < %s      | FileCheck %s --check-prefix=THUMB
// RUN: not llvm-mc -triple armv8a   -show-encoding -mattr=v8.5a < %s 2>&1 | FileCheck %s --check-prefix=NOMATMUL
// RUN: not llvm-mc -triple thumbv8a -show-encoding -mattr=v8.5a < %s 2>&1 | FileCheck %s --check-prefix=NOMATMUL

vsmmla.s8 q0, q1, q2
// ARM: vsmmla.s8       q0, q1, q2      @ encoding: [0x44,0x0c,0x22,0xfc]
// THUMB: vsmmla.s8       q0, q1, q2      @ encoding: [0x22,0xfc,0x44,0x0c]
// NOMATMUL: [[@LINE-3]]:{{[0-9]+}}: error: instruction requires: 8-bit integer matrix multiply

vummla.u8 q0, q1, q2
// ARM: vummla.u8       q0, q1, q2      @ encoding: [0x54,0x0c,0x22,0xfc]
// THUMB: vummla.u8       q0, q1, q2      @ encoding: [0x22,0xfc,0x54,0x0c]
// NOMATMUL: [[@LINE-3]]:{{[0-9]+}}: error: instruction requires: 8-bit integer matrix multiply

vusmmla.s8 q0, q1, q2
// ARM: vusmmla.s8      q0, q1, q2      @ encoding: [0x44,0x0c,0xa2,0xfc]
// THUMB: vusmmla.s8      q0, q1, q2      @ encoding: [0xa2,0xfc,0x44,0x0c]
// NOMATMUL: [[@LINE-3]]:{{[0-9]+}}: error: instruction requires: 8-bit integer matrix multiply

vusdot.s8 d0, d1, d2
// ARM: vusdot.s8       d0, d1, d2      @ encoding: [0x02,0x0d,0xa1,0xfc]
// THUMB: vusdot.s8       d0, d1, d2      @ encoding: [0xa1,0xfc,0x02,0x0d]
// NOMATMUL: [[@LINE-3]]:{{[0-9]+}}: error: instruction requires: 8-bit integer matrix multiply

vusdot.s8 q0, q1, q2
// ARM: vusdot.s8       q0, q1, q2      @ encoding: [0x44,0x0d,0xa2,0xfc]
// THUMB: vusdot.s8       q0, q1, q2      @ encoding: [0xa2,0xfc,0x44,0x0d]
// NOMATMUL: [[@LINE-3]]:{{[0-9]+}}: error: instruction requires: 8-bit integer matrix multiply

vusdot.s8 d0, d1, d2[0]
// ARM: vusdot.s8       d0, d1, d2[0]   @ encoding: [0x02,0x0d,0x81,0xfe]
// THUMB: vusdot.s8       d0, d1, d2[0]   @ encoding: [0x81,0xfe,0x02,0x0d]
// NOMATMUL: [[@LINE-3]]:{{[0-9]+}}: error: instruction requires: 8-bit integer matrix multiply

vsudot.u8 d0, d1, d2[1]
// ARM: vsudot.u8       d0, d1, d2[1]   @ encoding: [0x32,0x0d,0x81,0xfe]
// THUMB: vsudot.u8       d0, d1, d2[1]   @ encoding: [0x81,0xfe,0x32,0x0d]
// NOMATMUL: [[@LINE-3]]:{{[0-9]+}}: error: instruction requires: 8-bit integer matrix multiply

vusdot.s8 q0, q1, d2[0]
// ARM: vusdot.s8       q0, q1, d2[0]   @ encoding: [0x42,0x0d,0x82,0xfe]
// THUMB: vusdot.s8       q0, q1, d2[0]   @ encoding: [0x82,0xfe,0x42,0x0d]
// NOMATMUL: [[@LINE-3]]:{{[0-9]+}}: error: instruction requires: 8-bit integer matrix multiply

vsudot.u8 q0, q1, d2[1]
// ARM: vsudot.u8       q0, q1, d2[1]   @ encoding: [0x72,0x0d,0x82,0xfe]
// THUMB: vsudot.u8       q0, q1, d2[1]   @ encoding: [0x82,0xfe,0x72,0x0d]
// NOMATMUL: [[@LINE-3]]:{{[0-9]+}}: error: instruction requires: 8-bit integer matrix multiply
