// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// Multiple and single, 2 regs

// 16-bit elements

fmul    {z0.h-z1.h}, {z0.h-z1.h}, z0.h  // 11000001-01100000-11101000-00000000
// CHECK-INST: fmul    { z0.h, z1.h }, { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0xe8,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c160e800 <unknown>

fmul    {z20.h-z21.h}, {z10.h-z11.h}, z10.h  // 11000001-01110100-11101001-01010100
// CHECK-INST: fmul    { z20.h, z21.h }, { z10.h, z11.h }, z10.h
// CHECK-ENCODING: [0x54,0xe9,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c174e954 <unknown>

fmul    {z30.h-z31.h}, {z30.h-z31.h}, z15.h  // 11000001-01111110-11101011-11011110
// CHECK-INST: fmul    { z30.h, z31.h }, { z30.h, z31.h }, z15.h
// CHECK-ENCODING: [0xde,0xeb,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c17eebde <unknown>

// 32-bit elements

fmul    {z0.s-z1.s}, {z0.s-z1.s}, z0.s  // 11000001-10100000-11101000-00000000
// CHECK-INST: fmul    { z0.s, z1.s }, { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x00,0xe8,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1a0e800 <unknown>

fmul    {z20.s-z21.s}, {z10.s-z11.s}, z10.s  // 11000001-10110100-11101001-01010100
// CHECK-INST: fmul    { z20.s, z21.s }, { z10.s, z11.s }, z10.s
// CHECK-ENCODING: [0x54,0xe9,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1b4e954 <unknown>

fmul    {z30.s-z31.s}, {z30.s-z31.s}, z15.s  // 11000001-10111110-11101011-11011110
// CHECK-INST: fmul    { z30.s, z31.s }, { z30.s, z31.s }, z15.s
// CHECK-ENCODING: [0xde,0xeb,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1beebde <unknown>

// 64-bit elements

fmul    {z0.d-z1.d}, {z0.d-z1.d}, z0.d  // 11000001-11100000-11101000-00000000
// CHECK-INST: fmul    { z0.d, z1.d }, { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x00,0xe8,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1e0e800 <unknown>

fmul    {z20.d-z21.d}, {z10.d-z11.d}, z10.d  // 11000001-11110100-11101001-01010100
// CHECK-INST: fmul    { z20.d, z21.d }, { z10.d, z11.d }, z10.d
// CHECK-ENCODING: [0x54,0xe9,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1f4e954 <unknown>

fmul    {z30.d-z31.d}, {z30.d-z31.d}, z15.d  // 11000001-11111110-11101011-11011110
// CHECK-INST: fmul    { z30.d, z31.d }, { z30.d, z31.d }, z15.d
// CHECK-ENCODING: [0xde,0xeb,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1feebde <unknown>

// Multiple and single, 4 regs

// 16-bit elements

fmul    {z0.h-z3.h}, {z0.h-z3.h}, z0.h  // 11000001-01100001-11101000-00000000
// CHECK-INST: fmul    { z0.h - z3.h }, { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0xe8,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c161e800 <unknown>

fmul    {z20.h-z23.h}, {z8.h-z11.h}, z10.h  // 11000001-01110101-11101001-00010100
// CHECK-INST: fmul    { z20.h - z23.h }, { z8.h - z11.h }, z10.h
// CHECK-ENCODING: [0x14,0xe9,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c175e914 <unknown>

fmul    {z28.h-z31.h}, {z28.h-z31.h}, z15.h  // 11000001-01111111-11101011-10011100
// CHECK-INST: fmul    { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK-ENCODING: [0x9c,0xeb,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c17feb9c <unknown>

// 32-bit elements

fmul    {z0.s-z3.s}, {z0.s-z3.s}, z0.s  // 11000001-10100001-11101000-00000000
// CHECK-INST: fmul    { z0.s - z3.s }, { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x00,0xe8,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1a1e800 <unknown>

fmul    {z20.s-z23.s}, {z8.s-z11.s}, z10.s  // 11000001-10110101-11101001-00010100
// CHECK-INST: fmul    { z20.s - z23.s }, { z8.s - z11.s }, z10.s
// CHECK-ENCODING: [0x14,0xe9,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1b5e914 <unknown>

fmul    {z28.s-z31.s}, {z28.s-z31.s}, z15.s  // 11000001-10111111-11101011-10011100
// CHECK-INST: fmul    { z28.s - z31.s }, { z28.s - z31.s }, z15.s
// CHECK-ENCODING: [0x9c,0xeb,0xbf,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1bfeb9c <unknown>

// 64-bit elements

fmul    {z0.d-z3.d}, {z0.d-z3.d}, z0.d  // 11000001-11100001-11101000-00000000
// CHECK-INST: fmul    { z0.d - z3.d }, { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x00,0xe8,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1e1e800 <unknown>

fmul    {z20.d-z23.d}, {z8.d-z11.d}, z10.d  // 11000001-11110101-11101001-00010100
// CHECK-INST: fmul    { z20.d - z23.d }, { z8.d - z11.d }, z10.d
// CHECK-ENCODING: [0x14,0xe9,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1f5e914 <unknown>

fmul    {z28.d-z31.d}, {z28.d-z31.d}, z15.d  // 11000001-11111111-11101011-10011100
// CHECK-INST: fmul    { z28.d - z31.d }, { z28.d - z31.d }, z15.d
// CHECK-ENCODING: [0x9c,0xeb,0xff,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1ffeb9c <unknown>

// Multiple, 2 regs

// 16-bit elements

fmul    {z0.h-z1.h}, {z0.h-z1.h}, {z0.h-z1.h}  // 11000001-01100000-11100100-00000000
// CHECK-INST: fmul    { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0xe4,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c160e400 <unknown>

fmul    {z20.h-z21.h}, {z10.h-z11.h}, {z20.h-z21.h}  // 11000001-01110100-11100101-01010100
// CHECK-INST: fmul    { z20.h, z21.h }, { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x54,0xe5,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c174e554 <unknown>

fmul    {z30.h-z31.h}, {z30.h-z31.h}, {z30.h-z31.h}  // 11000001-01111110-11100111-11011110
// CHECK-INST: fmul    { z30.h, z31.h }, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xde,0xe7,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c17ee7de <unknown>

// 32-bit elememnts

fmul    {z0.s-z1.s}, {z0.s-z1.s}, {z0.s-z1.s}  // 11000001-10100000-11100100-00000000
// CHECK-INST: fmul    { z0.s, z1.s }, { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0xe4,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1a0e400 <unknown>

fmul    {z20.s-z21.s}, {z10.s-z11.s}, {z20.s-z21.s}  // 11000001-10110100-11100101-01010100
// CHECK-INST: fmul    { z20.s, z21.s }, { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x54,0xe5,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1b4e554 <unknown>

fmul    {z30.s-z31.s}, {z30.s-z31.s}, {z30.s-z31.s}  // 11000001-10111110-11100111-11011110
// CHECK-INST: fmul    { z30.s, z31.s }, { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xde,0xe7,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1bee7de <unknown>

// 64-bit elements

fmul    {z0.d-z1.d}, {z0.d-z1.d}, {z0.d-z1.d}  // 11000001-11100000-11100100-00000000
// CHECK-INST: fmul    { z0.d, z1.d }, { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0xe4,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1e0e400 <unknown>

fmul    {z20.d-z21.d}, {z10.d-z11.d}, {z20.d-z21.d}  // 11000001-11110100-11100101-01010100
// CHECK-INST: fmul    { z20.d, z21.d }, { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x54,0xe5,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1f4e554 <unknown>

fmul    {z30.d-z31.d}, {z30.d-z31.d}, {z30.d-z31.d}  // 11000001-11111110-11100111-11011110
// CHECK-INST: fmul    { z30.d, z31.d }, { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xde,0xe7,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1fee7de <unknown>

// Multiple, 4 regs

// 16-bit elements

fmul    {z0.h-z3.h}, {z0.h-z3.h}, {z0.h-z3.h}  // 11000001-01100001-11100100-00000000
// CHECK-INST: fmul    { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0xe4,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c161e400 <unknown>

fmul    {z20.h-z23.h}, {z8.h-z11.h}, {z20.h-z23.h}  // 11000001-01110101-11100101-00010100
// CHECK-INST: fmul    { z20.h - z23.h }, { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x14,0xe5,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c175e514 <unknown>

fmul    {z28.h-z31.h}, {z28.h-z31.h}, {z28.h-z31.h}  // 11000001-01111101-11100111-10011100
// CHECK-INST: fmul    { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9c,0xe7,0x7d,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c17de79c <unknown>

// 32-bit elements

fmul    {z0.s-z3.s}, {z0.s-z3.s}, {z0.s-z3.s}  // 11000001-10100001-11100100-00000000
// CHECK-INST: fmul    { z0.s - z3.s }, { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0xe4,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1a1e400 <unknown>

fmul    {z20.s-z23.s}, {z8.s-z11.s}, {z20.s-z23.s}  // 11000001-10110101-11100101-00010100
// CHECK-INST: fmul    { z20.s - z23.s }, { z8.s - z11.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x14,0xe5,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1b5e514 <unknown>

fmul    {z28.s-z31.s}, {z28.s-z31.s}, {z28.s-z31.s}  // 11000001-10111101-11100111-10011100
// CHECK-INST: fmul    { z28.s - z31.s }, { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9c,0xe7,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1bde79c <unknown>

// 64-bit elements

fmul    {z0.d-z3.d}, {z0.d-z3.d}, {z0.d-z3.d}  // 11000001-11100001-11100100-00000000
// CHECK-INST: fmul    { z0.d - z3.d }, { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0xe4,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1e1e400 <unknown>

fmul    {z20.d-z23.d}, {z8.d-z11.d}, {z20.d-z23.d}  // 11000001-11110101-11100101-00010100
// CHECK-INST: fmul    { z20.d - z23.d }, { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x14,0xe5,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1f5e514 <unknown>

fmul    {z28.d-z31.d}, {z28.d-z31.d}, {z28.d-z31.d}  // 11000001-11111101-11100111-10011100
// CHECK-INST: fmul    { z28.d - z31.d }, { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x9c,0xe7,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: c1fde79c <unknown>
