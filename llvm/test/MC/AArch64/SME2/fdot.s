// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


fdot    za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00010000-00001000
// CHECK-INST: fdot    za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501008 <unknown>

fdot    za.s[w8, 0], {z0.h, z1.h}, z0.h[0]  // 11000001-01010000-00010000-00001000
// CHECK-INST: fdot    za.s[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x10,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501008 <unknown>

fdot    za.s[w10, 5, vgx2], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01010101-01001101
// CHECK-INST: fdot    za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x4d,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155554d <unknown>

fdot    za.s[w10, 5], {z10.h, z11.h}, z5.h[1]  // 11000001-01010101-01010101-01001101
// CHECK-INST: fdot    za.s[w10, 5, vgx2], { z10.h, z11.h }, z5.h[1]
// CHECK-ENCODING: [0x4d,0x55,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155554d <unknown>

fdot    za.s[w11, 7, vgx2], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01111101-10001111
// CHECK-INST: fdot    za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0x8f,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587d8f <unknown>

fdot    za.s[w11, 7], {z12.h, z13.h}, z8.h[3]  // 11000001-01011000-01111101-10001111
// CHECK-INST: fdot    za.s[w11, 7, vgx2], { z12.h, z13.h }, z8.h[3]
// CHECK-ENCODING: [0x8f,0x7d,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1587d8f <unknown>

fdot    za.s[w11, 7, vgx2], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01111111-11001111
// CHECK-INST: fdot    za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xcf,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fcf <unknown>

fdot    za.s[w11, 7], {z30.h, z31.h}, z15.h[3]  // 11000001-01011111-01111111-11001111
// CHECK-INST: fdot    za.s[w11, 7, vgx2], { z30.h, z31.h }, z15.h[3]
// CHECK-ENCODING: [0xcf,0x7f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15f7fcf <unknown>

fdot    za.s[w8, 5, vgx2], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00011110-00001101
// CHECK-INST: fdot    za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x0d,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e0d <unknown>

fdot    za.s[w8, 5], {z16.h, z17.h}, z0.h[3]  // 11000001-01010000-00011110-00001101
// CHECK-INST: fdot    za.s[w8, 5, vgx2], { z16.h, z17.h }, z0.h[3]
// CHECK-ENCODING: [0x0d,0x1e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1501e0d <unknown>

fdot    za.s[w8, 1, vgx2], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00010100-00001001
// CHECK-INST: fdot    za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1409 <unknown>

fdot    za.s[w8, 1], {z0.h, z1.h}, z14.h[1]  // 11000001-01011110-00010100-00001001
// CHECK-INST: fdot    za.s[w8, 1, vgx2], { z0.h, z1.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x14,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1409 <unknown>

fdot    za.s[w10, 0, vgx2], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01010110-01001000
// CHECK-INST: fdot    za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x48,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545648 <unknown>

fdot    za.s[w10, 0], {z18.h, z19.h}, z4.h[1]  // 11000001-01010100-01010110-01001000
// CHECK-INST: fdot    za.s[w10, 0, vgx2], { z18.h, z19.h }, z4.h[1]
// CHECK-ENCODING: [0x48,0x56,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1545648 <unknown>

fdot    za.s[w8, 0, vgx2], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00011001-10001000
// CHECK-INST: fdot    za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0x88,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1521988 <unknown>

fdot    za.s[w8, 0], {z12.h, z13.h}, z2.h[2]  // 11000001-01010010-00011001-10001000
// CHECK-INST: fdot    za.s[w8, 0, vgx2], { z12.h, z13.h }, z2.h[2]
// CHECK-ENCODING: [0x88,0x19,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1521988 <unknown>

fdot    za.s[w10, 1, vgx2], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01011000-00001001
// CHECK-INST: fdot    za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x09,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5809 <unknown>

fdot    za.s[w10, 1], {z0.h, z1.h}, z10.h[2]  // 11000001-01011010-01011000-00001001
// CHECK-INST: fdot    za.s[w10, 1, vgx2], { z0.h, z1.h }, z10.h[2]
// CHECK-ENCODING: [0x09,0x58,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15a5809 <unknown>

fdot    za.s[w8, 5, vgx2], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00011010-11001101
// CHECK-INST: fdot    za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xcd,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1acd <unknown>

fdot    za.s[w8, 5], {z22.h, z23.h}, z14.h[2]  // 11000001-01011110-00011010-11001101
// CHECK-INST: fdot    za.s[w8, 5, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xcd,0x1a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e1acd <unknown>

fdot    za.s[w11, 2, vgx2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01110101-00001010
// CHECK-INST: fdot    za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151750a <unknown>

fdot    za.s[w11, 2], {z8.h, z9.h}, z1.h[1]  // 11000001-01010001-01110101-00001010
// CHECK-INST: fdot    za.s[w11, 2, vgx2], { z8.h, z9.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0x75,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151750a <unknown>

fdot    za.s[w9, 7, vgx2], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00111001-10001111
// CHECK-INST: fdot    za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0x8f,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b398f <unknown>

fdot    za.s[w9, 7], {z12.h, z13.h}, z11.h[2]  // 11000001-01011011-00111001-10001111
// CHECK-INST: fdot    za.s[w9, 7, vgx2], { z12.h, z13.h }, z11.h[2]
// CHECK-ENCODING: [0x8f,0x39,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15b398f <unknown>


fdot    za.s[w8, 0, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-01010000-10010000-00001000
// CHECK-INST: fdot    za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509008 <unknown>

fdot    za.s[w8, 0], {z0.h - z3.h}, z0.h[0]  // 11000001-01010000-10010000-00001000
// CHECK-INST: fdot    za.s[w8, 0, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x08,0x90,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509008 <unknown>

fdot    za.s[w10, 5, vgx4], {z8.h - z11.h}, z5.h[1]  // 11000001-01010101-11010101-00001101
// CHECK-INST: fdot    za.s[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x0d,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d50d <unknown>

fdot    za.s[w10, 5], {z8.h - z11.h}, z5.h[1]  // 11000001-01010101-11010101-00001101
// CHECK-INST: fdot    za.s[w10, 5, vgx4], { z8.h - z11.h }, z5.h[1]
// CHECK-ENCODING: [0x0d,0xd5,0x55,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c155d50d <unknown>

fdot    za.s[w11, 7, vgx4], {z12.h - z15.h}, z8.h[3]  // 11000001-01011000-11111101-10001111
// CHECK-INST: fdot    za.s[w11, 7, vgx4], { z12.h - z15.h }, z8.h[3]
// CHECK-ENCODING: [0x8f,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fd8f <unknown>

fdot    za.s[w11, 7], {z12.h - z15.h}, z8.h[3]  // 11000001-01011000-11111101-10001111
// CHECK-INST: fdot    za.s[w11, 7, vgx4], { z12.h - z15.h }, z8.h[3]
// CHECK-ENCODING: [0x8f,0xfd,0x58,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c158fd8f <unknown>

fdot    za.s[w11, 7, vgx4], {z28.h - z31.h}, z15.h[3]  // 11000001-01011111-11111111-10001111
// CHECK-INST: fdot    za.s[w11, 7, vgx4], { z28.h - z31.h }, z15.h[3]
// CHECK-ENCODING: [0x8f,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fff8f <unknown>

fdot    za.s[w11, 7], {z28.h - z31.h}, z15.h[3]  // 11000001-01011111-11111111-10001111
// CHECK-INST: fdot    za.s[w11, 7, vgx4], { z28.h - z31.h }, z15.h[3]
// CHECK-ENCODING: [0x8f,0xff,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15fff8f <unknown>

fdot    za.s[w8, 5, vgx4], {z16.h - z19.h}, z0.h[3]  // 11000001-01010000-10011110-00001101
// CHECK-INST: fdot    za.s[w8, 5, vgx4], { z16.h - z19.h }, z0.h[3]
// CHECK-ENCODING: [0x0d,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e0d <unknown>

fdot    za.s[w8, 5], {z16.h - z19.h}, z0.h[3]  // 11000001-01010000-10011110-00001101
// CHECK-INST: fdot    za.s[w8, 5, vgx4], { z16.h - z19.h }, z0.h[3]
// CHECK-ENCODING: [0x0d,0x9e,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1509e0d <unknown>

fdot    za.s[w8, 1, vgx4], {z0.h - z3.h}, z14.h[1]  // 11000001-01011110-10010100-00001001
// CHECK-INST: fdot    za.s[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9409 <unknown>

fdot    za.s[w8, 1], {z0.h - z3.h}, z14.h[1]  // 11000001-01011110-10010100-00001001
// CHECK-INST: fdot    za.s[w8, 1, vgx4], { z0.h - z3.h }, z14.h[1]
// CHECK-ENCODING: [0x09,0x94,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9409 <unknown>

fdot    za.s[w10, 0, vgx4], {z16.h - z19.h}, z4.h[1]  // 11000001-01010100-11010110-00001000
// CHECK-INST: fdot    za.s[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x08,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d608 <unknown>

fdot    za.s[w10, 0], {z16.h - z19.h}, z4.h[1]  // 11000001-01010100-11010110-00001000
// CHECK-INST: fdot    za.s[w10, 0, vgx4], { z16.h - z19.h }, z4.h[1]
// CHECK-ENCODING: [0x08,0xd6,0x54,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c154d608 <unknown>

fdot    za.s[w8, 0, vgx4], {z12.h - z15.h}, z2.h[2]  // 11000001-01010010-10011001-10001000
// CHECK-INST: fdot    za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h[2]
// CHECK-ENCODING: [0x88,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1529988 <unknown>

fdot    za.s[w8, 0], {z12.h - z15.h}, z2.h[2]  // 11000001-01010010-10011001-10001000
// CHECK-INST: fdot    za.s[w8, 0, vgx4], { z12.h - z15.h }, z2.h[2]
// CHECK-ENCODING: [0x88,0x99,0x52,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1529988 <unknown>

fdot    za.s[w10, 1, vgx4], {z0.h - z3.h}, z10.h[2]  // 11000001-01011010-11011000-00001001
// CHECK-INST: fdot    za.s[w10, 1, vgx4], { z0.h - z3.h }, z10.h[2]
// CHECK-ENCODING: [0x09,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad809 <unknown>

fdot    za.s[w10, 1], {z0.h - z3.h}, z10.h[2]  // 11000001-01011010-11011000-00001001
// CHECK-INST: fdot    za.s[w10, 1, vgx4], { z0.h - z3.h }, z10.h[2]
// CHECK-ENCODING: [0x09,0xd8,0x5a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15ad809 <unknown>

fdot    za.s[w8, 5, vgx4], {z20.h - z23.h}, z14.h[2]  // 11000001-01011110-10011010-10001101
// CHECK-INST: fdot    za.s[w8, 5, vgx4], { z20.h - z23.h }, z14.h[2]
// CHECK-ENCODING: [0x8d,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9a8d <unknown>

fdot    za.s[w8, 5], {z20.h - z23.h}, z14.h[2]  // 11000001-01011110-10011010-10001101
// CHECK-INST: fdot    za.s[w8, 5, vgx4], { z20.h - z23.h }, z14.h[2]
// CHECK-ENCODING: [0x8d,0x9a,0x5e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15e9a8d <unknown>

fdot    za.s[w11, 2, vgx4], {z8.h - z11.h}, z1.h[1]  // 11000001-01010001-11110101-00001010
// CHECK-INST: fdot    za.s[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f50a <unknown>

fdot    za.s[w11, 2], {z8.h - z11.h}, z1.h[1]  // 11000001-01010001-11110101-00001010
// CHECK-INST: fdot    za.s[w11, 2, vgx4], { z8.h - z11.h }, z1.h[1]
// CHECK-ENCODING: [0x0a,0xf5,0x51,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c151f50a <unknown>

fdot    za.s[w9, 7, vgx4], {z12.h - z15.h}, z11.h[2]  // 11000001-01011011-10111001-10001111
// CHECK-INST: fdot    za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h[2]
// CHECK-ENCODING: [0x8f,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb98f <unknown>

fdot    za.s[w9, 7], {z12.h - z15.h}, z11.h[2]  // 11000001-01011011-10111001-10001111
// CHECK-INST: fdot    za.s[w9, 7, vgx4], { z12.h - z15.h }, z11.h[2]
// CHECK-ENCODING: [0x8f,0xb9,0x5b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c15bb98f <unknown>

