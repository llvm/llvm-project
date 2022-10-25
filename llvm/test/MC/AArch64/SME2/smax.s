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


smax    {z0.h, z1.h}, {z0.h, z1.h}, z0.h  // 11000001-01100000-10100000-00000000
// CHECK-INST: smax    { z0.h, z1.h }, { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160a000 <unknown>

smax    {z20.h, z21.h}, {z20.h, z21.h}, z5.h  // 11000001-01100101-10100000-00010100
// CHECK-INST: smax    { z20.h, z21.h }, { z20.h, z21.h }, z5.h
// CHECK-ENCODING: [0x14,0xa0,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165a014 <unknown>

smax    {z22.h, z23.h}, {z22.h, z23.h}, z8.h  // 11000001-01101000-10100000-00010110
// CHECK-INST: smax    { z22.h, z23.h }, { z22.h, z23.h }, z8.h
// CHECK-ENCODING: [0x16,0xa0,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168a016 <unknown>

smax    {z30.h, z31.h}, {z30.h, z31.h}, z15.h  // 11000001-01101111-10100000-00011110
// CHECK-INST: smax    { z30.h, z31.h }, { z30.h, z31.h }, z15.h
// CHECK-ENCODING: [0x1e,0xa0,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16fa01e <unknown>


smax    {z0.h, z1.h}, {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-01100000-10110000-00000000
// CHECK-INST: smax    { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0xb0,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160b000 <unknown>

smax    {z20.h, z21.h}, {z20.h, z21.h}, {z20.h, z21.h}  // 11000001-01110100-10110000-00010100
// CHECK-INST: smax    { z20.h, z21.h }, { z20.h, z21.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x14,0xb0,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c174b014 <unknown>

smax    {z22.h, z23.h}, {z22.h, z23.h}, {z8.h, z9.h}  // 11000001-01101000-10110000-00010110
// CHECK-INST: smax    { z22.h, z23.h }, { z22.h, z23.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x16,0xb0,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168b016 <unknown>

smax    {z30.h, z31.h}, {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-01111110-10110000-00011110
// CHECK-INST: smax    { z30.h, z31.h }, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x1e,0xb0,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17eb01e <unknown>


smax    {z0.s, z1.s}, {z0.s, z1.s}, z0.s  // 11000001-10100000-10100000-00000000
// CHECK-INST: smax    { z0.s, z1.s }, { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0a000 <unknown>

smax    {z20.s, z21.s}, {z20.s, z21.s}, z5.s  // 11000001-10100101-10100000-00010100
// CHECK-INST: smax    { z20.s, z21.s }, { z20.s, z21.s }, z5.s
// CHECK-ENCODING: [0x14,0xa0,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5a014 <unknown>

smax    {z22.s, z23.s}, {z22.s, z23.s}, z8.s  // 11000001-10101000-10100000-00010110
// CHECK-INST: smax    { z22.s, z23.s }, { z22.s, z23.s }, z8.s
// CHECK-ENCODING: [0x16,0xa0,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8a016 <unknown>

smax    {z30.s, z31.s}, {z30.s, z31.s}, z15.s  // 11000001-10101111-10100000-00011110
// CHECK-INST: smax    { z30.s, z31.s }, { z30.s, z31.s }, z15.s
// CHECK-ENCODING: [0x1e,0xa0,0xaf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1afa01e <unknown>


smax    {z0.s, z1.s}, {z0.s, z1.s}, {z0.s, z1.s}  // 11000001-10100000-10110000-00000000
// CHECK-INST: smax    { z0.s, z1.s }, { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0xb0,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0b000 <unknown>

smax    {z20.s, z21.s}, {z20.s, z21.s}, {z20.s, z21.s}  // 11000001-10110100-10110000-00010100
// CHECK-INST: smax    { z20.s, z21.s }, { z20.s, z21.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x14,0xb0,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b4b014 <unknown>

smax    {z22.s, z23.s}, {z22.s, z23.s}, {z8.s, z9.s}  // 11000001-10101000-10110000-00010110
// CHECK-INST: smax    { z22.s, z23.s }, { z22.s, z23.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x16,0xb0,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8b016 <unknown>

smax    {z30.s, z31.s}, {z30.s, z31.s}, {z30.s, z31.s}  // 11000001-10111110-10110000-00011110
// CHECK-INST: smax    { z30.s, z31.s }, { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x1e,0xb0,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1beb01e <unknown>


smax    {z0.d, z1.d}, {z0.d, z1.d}, z0.d  // 11000001-11100000-10100000-00000000
// CHECK-INST: smax    { z0.d, z1.d }, { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0a000 <unknown>

smax    {z20.d, z21.d}, {z20.d, z21.d}, z5.d  // 11000001-11100101-10100000-00010100
// CHECK-INST: smax    { z20.d, z21.d }, { z20.d, z21.d }, z5.d
// CHECK-ENCODING: [0x14,0xa0,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5a014 <unknown>

smax    {z22.d, z23.d}, {z22.d, z23.d}, z8.d  // 11000001-11101000-10100000-00010110
// CHECK-INST: smax    { z22.d, z23.d }, { z22.d, z23.d }, z8.d
// CHECK-ENCODING: [0x16,0xa0,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8a016 <unknown>

smax    {z30.d, z31.d}, {z30.d, z31.d}, z15.d  // 11000001-11101111-10100000-00011110
// CHECK-INST: smax    { z30.d, z31.d }, { z30.d, z31.d }, z15.d
// CHECK-ENCODING: [0x1e,0xa0,0xef,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1efa01e <unknown>


smax    {z0.d, z1.d}, {z0.d, z1.d}, {z0.d, z1.d}  // 11000001-11100000-10110000-00000000
// CHECK-INST: smax    { z0.d, z1.d }, { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0xb0,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0b000 <unknown>

smax    {z20.d, z21.d}, {z20.d, z21.d}, {z20.d, z21.d}  // 11000001-11110100-10110000-00010100
// CHECK-INST: smax    { z20.d, z21.d }, { z20.d, z21.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x14,0xb0,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f4b014 <unknown>

smax    {z22.d, z23.d}, {z22.d, z23.d}, {z8.d, z9.d}  // 11000001-11101000-10110000-00010110
// CHECK-INST: smax    { z22.d, z23.d }, { z22.d, z23.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x16,0xb0,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8b016 <unknown>

smax    {z30.d, z31.d}, {z30.d, z31.d}, {z30.d, z31.d}  // 11000001-11111110-10110000-00011110
// CHECK-INST: smax    { z30.d, z31.d }, { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x1e,0xb0,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1feb01e <unknown>


smax    {z0.b, z1.b}, {z0.b, z1.b}, z0.b  // 11000001-00100000-10100000-00000000
// CHECK-INST: smax    { z0.b, z1.b }, { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120a000 <unknown>

smax    {z20.b, z21.b}, {z20.b, z21.b}, z5.b  // 11000001-00100101-10100000-00010100
// CHECK-INST: smax    { z20.b, z21.b }, { z20.b, z21.b }, z5.b
// CHECK-ENCODING: [0x14,0xa0,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125a014 <unknown>

smax    {z22.b, z23.b}, {z22.b, z23.b}, z8.b  // 11000001-00101000-10100000-00010110
// CHECK-INST: smax    { z22.b, z23.b }, { z22.b, z23.b }, z8.b
// CHECK-ENCODING: [0x16,0xa0,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128a016 <unknown>

smax    {z30.b, z31.b}, {z30.b, z31.b}, z15.b  // 11000001-00101111-10100000-00011110
// CHECK-INST: smax    { z30.b, z31.b }, { z30.b, z31.b }, z15.b
// CHECK-ENCODING: [0x1e,0xa0,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12fa01e <unknown>


smax    {z0.b, z1.b}, {z0.b, z1.b}, {z0.b, z1.b}  // 11000001-00100000-10110000-00000000
// CHECK-INST: smax    { z0.b, z1.b }, { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x00,0xb0,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120b000 <unknown>

smax    {z20.b, z21.b}, {z20.b, z21.b}, {z20.b, z21.b}  // 11000001-00110100-10110000-00010100
// CHECK-INST: smax    { z20.b, z21.b }, { z20.b, z21.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x14,0xb0,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c134b014 <unknown>

smax    {z22.b, z23.b}, {z22.b, z23.b}, {z8.b, z9.b}  // 11000001-00101000-10110000-00010110
// CHECK-INST: smax    { z22.b, z23.b }, { z22.b, z23.b }, { z8.b, z9.b }
// CHECK-ENCODING: [0x16,0xb0,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128b016 <unknown>

smax    {z30.b, z31.b}, {z30.b, z31.b}, {z30.b, z31.b}  // 11000001-00111110-10110000-00011110
// CHECK-INST: smax    { z30.b, z31.b }, { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0x1e,0xb0,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13eb01e <unknown>


smax    {z0.h - z3.h}, {z0.h - z3.h}, z0.h  // 11000001-01100000-10101000-00000000
// CHECK-INST: smax    { z0.h - z3.h }, { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0xa8,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160a800 <unknown>

smax    {z20.h - z23.h}, {z20.h - z23.h}, z5.h  // 11000001-01100101-10101000-00010100
// CHECK-INST: smax    { z20.h - z23.h }, { z20.h - z23.h }, z5.h
// CHECK-ENCODING: [0x14,0xa8,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165a814 <unknown>

smax    {z20.h - z23.h}, {z20.h - z23.h}, z8.h  // 11000001-01101000-10101000-00010100
// CHECK-INST: smax    { z20.h - z23.h }, { z20.h - z23.h }, z8.h
// CHECK-ENCODING: [0x14,0xa8,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168a814 <unknown>

smax    {z28.h - z31.h}, {z28.h - z31.h}, z15.h  // 11000001-01101111-10101000-00011100
// CHECK-INST: smax    { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK-ENCODING: [0x1c,0xa8,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16fa81c <unknown>


smax    {z0.h - z3.h}, {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-01100000-10111000-00000000
// CHECK-INST: smax    { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0xb8,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160b800 <unknown>

smax    {z20.h - z23.h}, {z20.h - z23.h}, {z20.h - z23.h}  // 11000001-01110100-10111000-00010100
// CHECK-INST: smax    { z20.h - z23.h }, { z20.h - z23.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x14,0xb8,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c174b814 <unknown>

smax    {z20.h - z23.h}, {z20.h - z23.h}, {z8.h - z11.h}  // 11000001-01101000-10111000-00010100
// CHECK-INST: smax    { z20.h - z23.h }, { z20.h - z23.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x14,0xb8,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168b814 <unknown>

smax    {z28.h - z31.h}, {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-01111100-10111000-00011100
// CHECK-INST: smax    { z28.h - z31.h }, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x1c,0xb8,0x7c,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17cb81c <unknown>


smax    {z0.s - z3.s}, {z0.s - z3.s}, z0.s  // 11000001-10100000-10101000-00000000
// CHECK-INST: smax    { z0.s - z3.s }, { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x00,0xa8,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0a800 <unknown>

smax    {z20.s - z23.s}, {z20.s - z23.s}, z5.s  // 11000001-10100101-10101000-00010100
// CHECK-INST: smax    { z20.s - z23.s }, { z20.s - z23.s }, z5.s
// CHECK-ENCODING: [0x14,0xa8,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5a814 <unknown>

smax    {z20.s - z23.s}, {z20.s - z23.s}, z8.s  // 11000001-10101000-10101000-00010100
// CHECK-INST: smax    { z20.s - z23.s }, { z20.s - z23.s }, z8.s
// CHECK-ENCODING: [0x14,0xa8,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8a814 <unknown>

smax    {z28.s - z31.s}, {z28.s - z31.s}, z15.s  // 11000001-10101111-10101000-00011100
// CHECK-INST: smax    { z28.s - z31.s }, { z28.s - z31.s }, z15.s
// CHECK-ENCODING: [0x1c,0xa8,0xaf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1afa81c <unknown>


smax    {z0.s - z3.s}, {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10100000-10111000-00000000
// CHECK-INST: smax    { z0.s - z3.s }, { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0xb8,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0b800 <unknown>

smax    {z20.s - z23.s}, {z20.s - z23.s}, {z20.s - z23.s}  // 11000001-10110100-10111000-00010100
// CHECK-INST: smax    { z20.s - z23.s }, { z20.s - z23.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x14,0xb8,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b4b814 <unknown>

smax    {z20.s - z23.s}, {z20.s - z23.s}, {z8.s - z11.s}  // 11000001-10101000-10111000-00010100
// CHECK-INST: smax    { z20.s - z23.s }, { z20.s - z23.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x14,0xb8,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8b814 <unknown>

smax    {z28.s - z31.s}, {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111100-10111000-00011100
// CHECK-INST: smax    { z28.s - z31.s }, { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x1c,0xb8,0xbc,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bcb81c <unknown>


smax    {z0.d - z3.d}, {z0.d - z3.d}, z0.d  // 11000001-11100000-10101000-00000000
// CHECK-INST: smax    { z0.d - z3.d }, { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x00,0xa8,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0a800 <unknown>

smax    {z20.d - z23.d}, {z20.d - z23.d}, z5.d  // 11000001-11100101-10101000-00010100
// CHECK-INST: smax    { z20.d - z23.d }, { z20.d - z23.d }, z5.d
// CHECK-ENCODING: [0x14,0xa8,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5a814 <unknown>

smax    {z20.d - z23.d}, {z20.d - z23.d}, z8.d  // 11000001-11101000-10101000-00010100
// CHECK-INST: smax    { z20.d - z23.d }, { z20.d - z23.d }, z8.d
// CHECK-ENCODING: [0x14,0xa8,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8a814 <unknown>

smax    {z28.d - z31.d}, {z28.d - z31.d}, z15.d  // 11000001-11101111-10101000-00011100
// CHECK-INST: smax    { z28.d - z31.d }, { z28.d - z31.d }, z15.d
// CHECK-ENCODING: [0x1c,0xa8,0xef,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1efa81c <unknown>


smax    {z0.d - z3.d}, {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100000-10111000-00000000
// CHECK-INST: smax    { z0.d - z3.d }, { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0xb8,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0b800 <unknown>

smax    {z20.d - z23.d}, {z20.d - z23.d}, {z20.d - z23.d}  // 11000001-11110100-10111000-00010100
// CHECK-INST: smax    { z20.d - z23.d }, { z20.d - z23.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x14,0xb8,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f4b814 <unknown>

smax    {z20.d - z23.d}, {z20.d - z23.d}, {z8.d - z11.d}  // 11000001-11101000-10111000-00010100
// CHECK-INST: smax    { z20.d - z23.d }, { z20.d - z23.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x14,0xb8,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8b814 <unknown>

smax    {z28.d - z31.d}, {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111100-10111000-00011100
// CHECK-INST: smax    { z28.d - z31.d }, { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x1c,0xb8,0xfc,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fcb81c <unknown>


smax    {z0.b - z3.b}, {z0.b - z3.b}, z0.b  // 11000001-00100000-10101000-00000000
// CHECK-INST: smax    { z0.b - z3.b }, { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x00,0xa8,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120a800 <unknown>

smax    {z20.b - z23.b}, {z20.b - z23.b}, z5.b  // 11000001-00100101-10101000-00010100
// CHECK-INST: smax    { z20.b - z23.b }, { z20.b - z23.b }, z5.b
// CHECK-ENCODING: [0x14,0xa8,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125a814 <unknown>

smax    {z20.b - z23.b}, {z20.b - z23.b}, z8.b  // 11000001-00101000-10101000-00010100
// CHECK-INST: smax    { z20.b - z23.b }, { z20.b - z23.b }, z8.b
// CHECK-ENCODING: [0x14,0xa8,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128a814 <unknown>

smax    {z28.b - z31.b}, {z28.b - z31.b}, z15.b  // 11000001-00101111-10101000-00011100
// CHECK-INST: smax    { z28.b - z31.b }, { z28.b - z31.b }, z15.b
// CHECK-ENCODING: [0x1c,0xa8,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12fa81c <unknown>


smax    {z0.b - z3.b}, {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-00100000-10111000-00000000
// CHECK-INST: smax    { z0.b - z3.b }, { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x00,0xb8,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120b800 <unknown>

smax    {z20.b - z23.b}, {z20.b - z23.b}, {z20.b - z23.b}  // 11000001-00110100-10111000-00010100
// CHECK-INST: smax    { z20.b - z23.b }, { z20.b - z23.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x14,0xb8,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c134b814 <unknown>

smax    {z20.b - z23.b}, {z20.b - z23.b}, {z8.b - z11.b}  // 11000001-00101000-10111000-00010100
// CHECK-INST: smax    { z20.b - z23.b }, { z20.b - z23.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x14,0xb8,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128b814 <unknown>

smax    {z28.b - z31.b}, {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-00111100-10111000-00011100
// CHECK-INST: smax    { z28.b - z31.b }, { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x1c,0xb8,0x3c,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13cb81c <unknown>

