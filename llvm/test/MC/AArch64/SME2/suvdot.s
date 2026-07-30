// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


suvdot  za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10000000-00111000
// CHECK-INST: suvdot  za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x38,0x80,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508038 <unknown>

suvdot  za.s[w8, 0], {z0.b - z3.b}, z0.b[0]  // 11000001-01010000-10000000-00111000
// CHECK-INST: suvdot  za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x38,0x80,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508038 <unknown>

suvdot  za.s[w10, 5, vgx4], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11000101-00111101
// CHECK-INST: suvdot  za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x3d,0xc5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155c53d <unknown>

suvdot  za.s[w10, 5], {z8.b - z11.b}, z5.b[1]  // 11000001-01010101-11000101-00111101
// CHECK-INST: suvdot  za.s[w10, 5, vgx4], { z8.b - z11.b }, z5.b[1]
// CHECK-ENCODING: [0x3d,0xc5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155c53d <unknown>

suvdot  za.s[w11, 7, vgx4], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11101101-10111111
// CHECK-INST: suvdot  za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xbf,0xed,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158edbf <unknown>

suvdot  za.s[w11, 7], {z12.b - z15.b}, z8.b[3]  // 11000001-01011000-11101101-10111111
// CHECK-INST: suvdot  za.s[w11, 7, vgx4], { z12.b - z15.b }, z8.b[3]
// CHECK-ENCODING: [0xbf,0xed,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158edbf <unknown>

suvdot  za.s[w11, 7, vgx4], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11101111-10111111
// CHECK-INST: suvdot  za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xbf,0xef,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fefbf <unknown>

suvdot  za.s[w11, 7], {z28.b - z31.b}, z15.b[3]  // 11000001-01011111-11101111-10111111
// CHECK-INST: suvdot  za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0xbf,0xef,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fefbf <unknown>

suvdot  za.s[w8, 5, vgx4], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10001110-00111101
// CHECK-INST: suvdot  za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x3d,0x8e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508e3d <unknown>

suvdot  za.s[w8, 5], {z16.b - z19.b}, z0.b[3]  // 11000001-01010000-10001110-00111101
// CHECK-INST: suvdot  za.s[w8, 5, vgx4], { z16.b - z19.b }, z0.b[3]
// CHECK-ENCODING: [0x3d,0x8e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1508e3d <unknown>

suvdot  za.s[w8, 1, vgx4], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10000100-00111001
// CHECK-INST: suvdot  za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x39,0x84,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8439 <unknown>

suvdot  za.s[w8, 1], {z0.b - z3.b}, z14.b[1]  // 11000001-01011110-10000100-00111001
// CHECK-INST: suvdot  za.s[w8, 1, vgx4], { z0.b - z3.b }, z14.b[1]
// CHECK-ENCODING: [0x39,0x84,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8439 <unknown>

suvdot  za.s[w10, 0, vgx4], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11000110-00111000
// CHECK-INST: suvdot  za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x38,0xc6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154c638 <unknown>

suvdot  za.s[w10, 0], {z16.b - z19.b}, z4.b[1]  // 11000001-01010100-11000110-00111000
// CHECK-INST: suvdot  za.s[w10, 0, vgx4], { z16.b - z19.b }, z4.b[1]
// CHECK-ENCODING: [0x38,0xc6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154c638 <unknown>

suvdot  za.s[w8, 0, vgx4], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10001001-10111000
// CHECK-INST: suvdot  za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xb8,0x89,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15289b8 <unknown>

suvdot  za.s[w8, 0], {z12.b - z15.b}, z2.b[2]  // 11000001-01010010-10001001-10111000
// CHECK-INST: suvdot  za.s[w8, 0, vgx4], { z12.b - z15.b }, z2.b[2]
// CHECK-ENCODING: [0xb8,0x89,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15289b8 <unknown>

suvdot  za.s[w10, 1, vgx4], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11001000-00111001
// CHECK-INST: suvdot  za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x39,0xc8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ac839 <unknown>

suvdot  za.s[w10, 1], {z0.b - z3.b}, z10.b[2]  // 11000001-01011010-11001000-00111001
// CHECK-INST: suvdot  za.s[w10, 1, vgx4], { z0.b - z3.b }, z10.b[2]
// CHECK-ENCODING: [0x39,0xc8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ac839 <unknown>

suvdot  za.s[w8, 5, vgx4], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10001010-10111101
// CHECK-INST: suvdot  za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xbd,0x8a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8abd <unknown>

suvdot  za.s[w8, 5], {z20.b - z23.b}, z14.b[2]  // 11000001-01011110-10001010-10111101
// CHECK-INST: suvdot  za.s[w8, 5, vgx4], { z20.b - z23.b }, z14.b[2]
// CHECK-ENCODING: [0xbd,0x8a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e8abd <unknown>

suvdot  za.s[w11, 2, vgx4], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11100101-00111010
// CHECK-INST: suvdot  za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x3a,0xe5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151e53a <unknown>

suvdot  za.s[w11, 2], {z8.b - z11.b}, z1.b[1]  // 11000001-01010001-11100101-00111010
// CHECK-INST: suvdot  za.s[w11, 2, vgx4], { z8.b - z11.b }, z1.b[1]
// CHECK-ENCODING: [0x3a,0xe5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151e53a <unknown>

suvdot  za.s[w9, 7, vgx4], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10101001-10111111
// CHECK-INST: suvdot  za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xbf,0xa9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ba9bf <unknown>

suvdot  za.s[w9, 7], {z12.b - z15.b}, z11.b[2]  // 11000001-01011011-10101001-10111111
// CHECK-INST: suvdot  za.s[w9, 7, vgx4], { z12.b - z15.b }, z11.b[2]
// CHECK-ENCODING: [0xbf,0xa9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ba9bf <unknown>

