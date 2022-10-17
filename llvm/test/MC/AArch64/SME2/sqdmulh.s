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


sqdmulh {z0.h - z1.h}, {z0.h - z1.h}, z0.h  // 11000001-01100000-10100100-00000000
// CHECK-INST: sqdmulh { z0.h, z1.h }, { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x00,0xa4,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160a400 <unknown>

sqdmulh {z20.h - z21.h}, {z20.h - z21.h}, z5.h  // 11000001-01100101-10100100-00010100
// CHECK-INST: sqdmulh { z20.h, z21.h }, { z20.h, z21.h }, z5.h
// CHECK-ENCODING: [0x14,0xa4,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165a414 <unknown>

sqdmulh {z22.h - z23.h}, {z22.h - z23.h}, z8.h  // 11000001-01101000-10100100-00010110
// CHECK-INST: sqdmulh { z22.h, z23.h }, { z22.h, z23.h }, z8.h
// CHECK-ENCODING: [0x16,0xa4,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168a416 <unknown>

sqdmulh {z30.h - z31.h}, {z30.h - z31.h}, z15.h  // 11000001-01101111-10100100-00011110
// CHECK-INST: sqdmulh { z30.h, z31.h }, { z30.h, z31.h }, z15.h
// CHECK-ENCODING: [0x1e,0xa4,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16fa41e <unknown>


sqdmulh {z0.s - z1.s}, {z0.s - z1.s}, z0.s  // 11000001-10100000-10100100-00000000
// CHECK-INST: sqdmulh { z0.s, z1.s }, { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x00,0xa4,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0a400 <unknown>

sqdmulh {z20.s - z21.s}, {z20.s - z21.s}, z5.s  // 11000001-10100101-10100100-00010100
// CHECK-INST: sqdmulh { z20.s, z21.s }, { z20.s, z21.s }, z5.s
// CHECK-ENCODING: [0x14,0xa4,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5a414 <unknown>

sqdmulh {z22.s - z23.s}, {z22.s - z23.s}, z8.s  // 11000001-10101000-10100100-00010110
// CHECK-INST: sqdmulh { z22.s, z23.s }, { z22.s, z23.s }, z8.s
// CHECK-ENCODING: [0x16,0xa4,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8a416 <unknown>

sqdmulh {z30.s - z31.s}, {z30.s - z31.s}, z15.s  // 11000001-10101111-10100100-00011110
// CHECK-INST: sqdmulh { z30.s, z31.s }, { z30.s, z31.s }, z15.s
// CHECK-ENCODING: [0x1e,0xa4,0xaf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1afa41e <unknown>


sqdmulh {z0.d - z1.d}, {z0.d - z1.d}, z0.d  // 11000001-11100000-10100100-00000000
// CHECK-INST: sqdmulh { z0.d, z1.d }, { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x00,0xa4,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0a400 <unknown>

sqdmulh {z20.d - z21.d}, {z20.d - z21.d}, z5.d  // 11000001-11100101-10100100-00010100
// CHECK-INST: sqdmulh { z20.d, z21.d }, { z20.d, z21.d }, z5.d
// CHECK-ENCODING: [0x14,0xa4,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5a414 <unknown>

sqdmulh {z22.d - z23.d}, {z22.d - z23.d}, z8.d  // 11000001-11101000-10100100-00010110
// CHECK-INST: sqdmulh { z22.d, z23.d }, { z22.d, z23.d }, z8.d
// CHECK-ENCODING: [0x16,0xa4,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8a416 <unknown>

sqdmulh {z30.d - z31.d}, {z30.d - z31.d}, z15.d  // 11000001-11101111-10100100-00011110
// CHECK-INST: sqdmulh { z30.d, z31.d }, { z30.d, z31.d }, z15.d
// CHECK-ENCODING: [0x1e,0xa4,0xef,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1efa41e <unknown>


sqdmulh {z0.b - z1.b}, {z0.b - z1.b}, z0.b  // 11000001-00100000-10100100-00000000
// CHECK-INST: sqdmulh { z0.b, z1.b }, { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x00,0xa4,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120a400 <unknown>

sqdmulh {z20.b - z21.b}, {z20.b - z21.b}, z5.b  // 11000001-00100101-10100100-00010100
// CHECK-INST: sqdmulh { z20.b, z21.b }, { z20.b, z21.b }, z5.b
// CHECK-ENCODING: [0x14,0xa4,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125a414 <unknown>

sqdmulh {z22.b - z23.b}, {z22.b - z23.b}, z8.b  // 11000001-00101000-10100100-00010110
// CHECK-INST: sqdmulh { z22.b, z23.b }, { z22.b, z23.b }, z8.b
// CHECK-ENCODING: [0x16,0xa4,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128a416 <unknown>

sqdmulh {z30.b - z31.b}, {z30.b - z31.b}, z15.b  // 11000001-00101111-10100100-00011110
// CHECK-INST: sqdmulh { z30.b, z31.b }, { z30.b, z31.b }, z15.b
// CHECK-ENCODING: [0x1e,0xa4,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12fa41e <unknown>


sqdmulh {z0.h - z3.h}, {z0.h - z3.h}, z0.h  // 11000001-01100000-10101100-00000000
// CHECK-INST: sqdmulh { z0.h - z3.h }, { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x00,0xac,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160ac00 <unknown>

sqdmulh {z20.h - z23.h}, {z20.h - z23.h}, z5.h  // 11000001-01100101-10101100-00010100
// CHECK-INST: sqdmulh { z20.h - z23.h }, { z20.h - z23.h }, z5.h
// CHECK-ENCODING: [0x14,0xac,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c165ac14 <unknown>

sqdmulh {z20.h - z23.h}, {z20.h - z23.h}, z8.h  // 11000001-01101000-10101100-00010100
// CHECK-INST: sqdmulh { z20.h - z23.h }, { z20.h - z23.h }, z8.h
// CHECK-ENCODING: [0x14,0xac,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168ac14 <unknown>

sqdmulh {z28.h - z31.h}, {z28.h - z31.h}, z15.h  // 11000001-01101111-10101100-00011100
// CHECK-INST: sqdmulh { z28.h - z31.h }, { z28.h - z31.h }, z15.h
// CHECK-ENCODING: [0x1c,0xac,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16fac1c <unknown>


sqdmulh {z0.s - z3.s}, {z0.s - z3.s}, z0.s  // 11000001-10100000-10101100-00000000
// CHECK-INST: sqdmulh { z0.s - z3.s }, { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x00,0xac,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0ac00 <unknown>

sqdmulh {z20.s - z23.s}, {z20.s - z23.s}, z5.s  // 11000001-10100101-10101100-00010100
// CHECK-INST: sqdmulh { z20.s - z23.s }, { z20.s - z23.s }, z5.s
// CHECK-ENCODING: [0x14,0xac,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a5ac14 <unknown>

sqdmulh {z20.s - z23.s}, {z20.s - z23.s}, z8.s  // 11000001-10101000-10101100-00010100
// CHECK-INST: sqdmulh { z20.s - z23.s }, { z20.s - z23.s }, z8.s
// CHECK-ENCODING: [0x14,0xac,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a8ac14 <unknown>

sqdmulh {z28.s - z31.s}, {z28.s - z31.s}, z15.s  // 11000001-10101111-10101100-00011100
// CHECK-INST: sqdmulh { z28.s - z31.s }, { z28.s - z31.s }, z15.s
// CHECK-ENCODING: [0x1c,0xac,0xaf,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1afac1c <unknown>


sqdmulh {z0.d - z3.d}, {z0.d - z3.d}, z0.d  // 11000001-11100000-10101100-00000000
// CHECK-INST: sqdmulh { z0.d - z3.d }, { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x00,0xac,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0ac00 <unknown>

sqdmulh {z20.d - z23.d}, {z20.d - z23.d}, z5.d  // 11000001-11100101-10101100-00010100
// CHECK-INST: sqdmulh { z20.d - z23.d }, { z20.d - z23.d }, z5.d
// CHECK-ENCODING: [0x14,0xac,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5ac14 <unknown>

sqdmulh {z20.d - z23.d}, {z20.d - z23.d}, z8.d  // 11000001-11101000-10101100-00010100
// CHECK-INST: sqdmulh { z20.d - z23.d }, { z20.d - z23.d }, z8.d
// CHECK-ENCODING: [0x14,0xac,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8ac14 <unknown>

sqdmulh {z28.d - z31.d}, {z28.d - z31.d}, z15.d  // 11000001-11101111-10101100-00011100
// CHECK-INST: sqdmulh { z28.d - z31.d }, { z28.d - z31.d }, z15.d
// CHECK-ENCODING: [0x1c,0xac,0xef,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1efac1c <unknown>


sqdmulh {z0.b - z3.b}, {z0.b - z3.b}, z0.b  // 11000001-00100000-10101100-00000000
// CHECK-INST: sqdmulh { z0.b - z3.b }, { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x00,0xac,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c120ac00 <unknown>

sqdmulh {z20.b - z23.b}, {z20.b - z23.b}, z5.b  // 11000001-00100101-10101100-00010100
// CHECK-INST: sqdmulh { z20.b - z23.b }, { z20.b - z23.b }, z5.b
// CHECK-ENCODING: [0x14,0xac,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c125ac14 <unknown>

sqdmulh {z20.b - z23.b}, {z20.b - z23.b}, z8.b  // 11000001-00101000-10101100-00010100
// CHECK-INST: sqdmulh { z20.b - z23.b }, { z20.b - z23.b }, z8.b
// CHECK-ENCODING: [0x14,0xac,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c128ac14 <unknown>

sqdmulh {z28.b - z31.b}, {z28.b - z31.b}, z15.b  // 11000001-00101111-10101100-00011100
// CHECK-INST: sqdmulh { z28.b - z31.b }, { z28.b - z31.b }, z15.b
// CHECK-ENCODING: [0x1c,0xac,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12fac1c <unknown>
