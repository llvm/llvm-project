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


fvdot   za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00000000-00001000
// CHECK-INST: fvdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x00,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500008 <unknown>

fvdot   za.s[w8, 0], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00000000-00001000
// CHECK-INST: fvdot   za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x00,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500008 <unknown>

fvdot   za.s[w10, 5, vgx2], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01000101-01001101
// CHECK-INST: fvdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x4d,0x45,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155454d <unknown>

fvdot   za.s[w10, 5], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01000101-01001101
// CHECK-INST: fvdot   za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x4d,0x45,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155454d <unknown>

fvdot   za.s[w11, 7, vgx2], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01101101-10001111
// CHECK-INST: fvdot   za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0x8f,0x6d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1586d8f <unknown>

fvdot   za.s[w11, 7], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01101101-10001111
// CHECK-INST: fvdot   za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0x8f,0x6d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1586d8f <unknown>

fvdot   za.s[w11, 7, vgx2], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01101111-11001111
// CHECK-INST: fvdot   za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xcf,0x6f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f6fcf <unknown>

fvdot   za.s[w11, 7], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01101111-11001111
// CHECK-INST: fvdot   za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xcf,0x6f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f6fcf <unknown>

fvdot   za.s[w8, 5, vgx2], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00001110-00001101
// CHECK-INST: fvdot   za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x0d,0x0e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500e0d <unknown>

fvdot   za.s[w8, 5], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00001110-00001101
// CHECK-INST: fvdot   za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x0d,0x0e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1500e0d <unknown>

fvdot   za.s[w8, 1, vgx2], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00000100-00001001
// CHECK-INST: fvdot   za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x04,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0409 <unknown>

fvdot   za.s[w8, 1], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00000100-00001001
// CHECK-INST: fvdot   za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x04,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0409 <unknown>

fvdot   za.s[w10, 0, vgx2], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01000110-01001000
// CHECK-INST: fvdot   za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x48,0x46,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1544648 <unknown>

fvdot   za.s[w10, 0], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01000110-01001000
// CHECK-INST: fvdot   za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x48,0x46,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1544648 <unknown>

fvdot   za.s[w8, 0, vgx2], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00001001-10001000
// CHECK-INST: fvdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0x88,0x09,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1520988 <unknown>

fvdot   za.s[w8, 0], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00001001-10001000
// CHECK-INST: fvdot   za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0x88,0x09,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1520988 <unknown>

fvdot   za.s[w10, 1, vgx2], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01001000-00001001
// CHECK-INST: fvdot   za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x09,0x48,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a4809 <unknown>

fvdot   za.s[w10, 1], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01001000-00001001
// CHECK-INST: fvdot   za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x09,0x48,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a4809 <unknown>

fvdot   za.s[w8, 5, vgx2], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00001010-11001101
// CHECK-INST: fvdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xcd,0x0a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0acd <unknown>

fvdot   za.s[w8, 5], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00001010-11001101
// CHECK-INST: fvdot   za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xcd,0x0a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e0acd <unknown>

fvdot   za.s[w11, 2, vgx2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01100101-00001010
// CHECK-INST: fvdot   za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0x65,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151650a <unknown>

fvdot   za.s[w11, 2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01100101-00001010
// CHECK-INST: fvdot   za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0x65,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151650a <unknown>

fvdot   za.s[w9, 7, vgx2], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00101001-10001111
// CHECK-INST: fvdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0x8f,0x29,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b298f <unknown>

fvdot   za.s[w9, 7], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00101001-10001111
// CHECK-INST: fvdot   za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0x8f,0x29,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b298f <unknown>

