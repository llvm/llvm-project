// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sme-f64f64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sme-f64f64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


fadd    za.d[w8, 0, vgx2], {z0.d, z1.d}  // 11000001-11100000-00011100-00000000
// CHECK-INST: fadd    za.d[w8, 0, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e01c00 <unknown>

fadd    za.d[w8, 0], {z0.d, z1.d}  // 11000001-11100000-00011100-00000000
// CHECK-INST: fadd    za.d[w8, 0, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e01c00 <unknown>

fadd    za.d[w10, 5, vgx2], {z10.d, z11.d}  // 11000001-11100000-01011101-01000101
// CHECK-INST: fadd    za.d[w10, 5, vgx2], { z10.d, z11.d }
// CHECK-ENCODING: [0x45,0x5d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e05d45 <unknown>

fadd    za.d[w10, 5], {z10.d, z11.d}  // 11000001-11100000-01011101-01000101
// CHECK-INST: fadd    za.d[w10, 5, vgx2], { z10.d, z11.d }
// CHECK-ENCODING: [0x45,0x5d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e05d45 <unknown>

fadd    za.d[w11, 7, vgx2], {z12.d, z13.d}  // 11000001-11100000-01111101-10000111
// CHECK-INST: fadd    za.d[w11, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e07d87 <unknown>

fadd    za.d[w11, 7], {z12.d, z13.d}  // 11000001-11100000-01111101-10000111
// CHECK-INST: fadd    za.d[w11, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e07d87 <unknown>

fadd    za.d[w11, 7, vgx2], {z30.d, z31.d}  // 11000001-11100000-01111111-11000111
// CHECK-INST: fadd    za.d[w11, 7, vgx2], { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0x7f,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e07fc7 <unknown>

fadd    za.d[w11, 7], {z30.d, z31.d}  // 11000001-11100000-01111111-11000111
// CHECK-INST: fadd    za.d[w11, 7, vgx2], { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0x7f,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e07fc7 <unknown>

fadd    za.d[w8, 5, vgx2], {z16.d, z17.d}  // 11000001-11100000-00011110-00000101
// CHECK-INST: fadd    za.d[w8, 5, vgx2], { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e01e05 <unknown>

fadd    za.d[w8, 5], {z16.d, z17.d}  // 11000001-11100000-00011110-00000101
// CHECK-INST: fadd    za.d[w8, 5, vgx2], { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e01e05 <unknown>

fadd    za.d[w8, 1, vgx2], {z0.d, z1.d}  // 11000001-11100000-00011100-00000001
// CHECK-INST: fadd    za.d[w8, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e01c01 <unknown>

fadd    za.d[w8, 1], {z0.d, z1.d}  // 11000001-11100000-00011100-00000001
// CHECK-INST: fadd    za.d[w8, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x1c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e01c01 <unknown>

fadd    za.d[w10, 0, vgx2], {z18.d, z19.d}  // 11000001-11100000-01011110-01000000
// CHECK-INST: fadd    za.d[w10, 0, vgx2], { z18.d, z19.d }
// CHECK-ENCODING: [0x40,0x5e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e05e40 <unknown>

fadd    za.d[w10, 0], {z18.d, z19.d}  // 11000001-11100000-01011110-01000000
// CHECK-INST: fadd    za.d[w10, 0, vgx2], { z18.d, z19.d }
// CHECK-ENCODING: [0x40,0x5e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e05e40 <unknown>

fadd    za.d[w8, 0, vgx2], {z12.d, z13.d}  // 11000001-11100000-00011101-10000000
// CHECK-INST: fadd    za.d[w8, 0, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x80,0x1d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e01d80 <unknown>

fadd    za.d[w8, 0], {z12.d, z13.d}  // 11000001-11100000-00011101-10000000
// CHECK-INST: fadd    za.d[w8, 0, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x80,0x1d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e01d80 <unknown>

fadd    za.d[w10, 1, vgx2], {z0.d, z1.d}  // 11000001-11100000-01011100-00000001
// CHECK-INST: fadd    za.d[w10, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x5c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e05c01 <unknown>

fadd    za.d[w10, 1], {z0.d, z1.d}  // 11000001-11100000-01011100-00000001
// CHECK-INST: fadd    za.d[w10, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x5c,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e05c01 <unknown>

fadd    za.d[w8, 5, vgx2], {z22.d, z23.d}  // 11000001-11100000-00011110-11000101
// CHECK-INST: fadd    za.d[w8, 5, vgx2], { z22.d, z23.d }
// CHECK-ENCODING: [0xc5,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e01ec5 <unknown>

fadd    za.d[w8, 5], {z22.d, z23.d}  // 11000001-11100000-00011110-11000101
// CHECK-INST: fadd    za.d[w8, 5, vgx2], { z22.d, z23.d }
// CHECK-ENCODING: [0xc5,0x1e,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e01ec5 <unknown>

fadd    za.d[w11, 2, vgx2], {z8.d, z9.d}  // 11000001-11100000-01111101-00000010
// CHECK-INST: fadd    za.d[w11, 2, vgx2], { z8.d, z9.d }
// CHECK-ENCODING: [0x02,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e07d02 <unknown>

fadd    za.d[w11, 2], {z8.d, z9.d}  // 11000001-11100000-01111101-00000010
// CHECK-INST: fadd    za.d[w11, 2, vgx2], { z8.d, z9.d }
// CHECK-ENCODING: [0x02,0x7d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e07d02 <unknown>

fadd    za.d[w9, 7, vgx2], {z12.d, z13.d}  // 11000001-11100000-00111101-10000111
// CHECK-INST: fadd    za.d[w9, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x3d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e03d87 <unknown>

fadd    za.d[w9, 7], {z12.d, z13.d}  // 11000001-11100000-00111101-10000111
// CHECK-INST: fadd    za.d[w9, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x3d,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e03d87 <unknown>


fadd    za.s[w8, 0, vgx2], {z0.s, z1.s}  // 11000001-10100000-00011100-00000000
// CHECK-INST: fadd    za.s[w8, 0, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c00 <unknown>

fadd    za.s[w8, 0], {z0.s, z1.s}  // 11000001-10100000-00011100-00000000
// CHECK-INST: fadd    za.s[w8, 0, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c00 <unknown>

fadd    za.s[w10, 5, vgx2], {z10.s, z11.s}  // 11000001-10100000-01011101-01000101
// CHECK-INST: fadd    za.s[w10, 5, vgx2], { z10.s, z11.s }
// CHECK-ENCODING: [0x45,0x5d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05d45 <unknown>

fadd    za.s[w10, 5], {z10.s, z11.s}  // 11000001-10100000-01011101-01000101
// CHECK-INST: fadd    za.s[w10, 5, vgx2], { z10.s, z11.s }
// CHECK-ENCODING: [0x45,0x5d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05d45 <unknown>

fadd    za.s[w11, 7, vgx2], {z12.s, z13.s}  // 11000001-10100000-01111101-10000111
// CHECK-INST: fadd    za.s[w11, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d87 <unknown>

fadd    za.s[w11, 7], {z12.s, z13.s}  // 11000001-10100000-01111101-10000111
// CHECK-INST: fadd    za.s[w11, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d87 <unknown>

fadd    za.s[w11, 7, vgx2], {z30.s, z31.s}  // 11000001-10100000-01111111-11000111
// CHECK-INST: fadd    za.s[w11, 7, vgx2], { z30.s, z31.s }
// CHECK-ENCODING: [0xc7,0x7f,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07fc7 <unknown>

fadd    za.s[w11, 7], {z30.s, z31.s}  // 11000001-10100000-01111111-11000111
// CHECK-INST: fadd    za.s[w11, 7, vgx2], { z30.s, z31.s }
// CHECK-ENCODING: [0xc7,0x7f,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07fc7 <unknown>

fadd    za.s[w8, 5, vgx2], {z16.s, z17.s}  // 11000001-10100000-00011110-00000101
// CHECK-INST: fadd    za.s[w8, 5, vgx2], { z16.s, z17.s }
// CHECK-ENCODING: [0x05,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01e05 <unknown>

fadd    za.s[w8, 5], {z16.s, z17.s}  // 11000001-10100000-00011110-00000101
// CHECK-INST: fadd    za.s[w8, 5, vgx2], { z16.s, z17.s }
// CHECK-ENCODING: [0x05,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01e05 <unknown>

fadd    za.s[w8, 1, vgx2], {z0.s, z1.s}  // 11000001-10100000-00011100-00000001
// CHECK-INST: fadd    za.s[w8, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c01 <unknown>

fadd    za.s[w8, 1], {z0.s, z1.s}  // 11000001-10100000-00011100-00000001
// CHECK-INST: fadd    za.s[w8, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0x1c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01c01 <unknown>

fadd    za.s[w10, 0, vgx2], {z18.s, z19.s}  // 11000001-10100000-01011110-01000000
// CHECK-INST: fadd    za.s[w10, 0, vgx2], { z18.s, z19.s }
// CHECK-ENCODING: [0x40,0x5e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05e40 <unknown>

fadd    za.s[w10, 0], {z18.s, z19.s}  // 11000001-10100000-01011110-01000000
// CHECK-INST: fadd    za.s[w10, 0, vgx2], { z18.s, z19.s }
// CHECK-ENCODING: [0x40,0x5e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05e40 <unknown>

fadd    za.s[w8, 0, vgx2], {z12.s, z13.s}  // 11000001-10100000-00011101-10000000
// CHECK-INST: fadd    za.s[w8, 0, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x80,0x1d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01d80 <unknown>

fadd    za.s[w8, 0], {z12.s, z13.s}  // 11000001-10100000-00011101-10000000
// CHECK-INST: fadd    za.s[w8, 0, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x80,0x1d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01d80 <unknown>

fadd    za.s[w10, 1, vgx2], {z0.s, z1.s}  // 11000001-10100000-01011100-00000001
// CHECK-INST: fadd    za.s[w10, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0x5c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05c01 <unknown>

fadd    za.s[w10, 1], {z0.s, z1.s}  // 11000001-10100000-01011100-00000001
// CHECK-INST: fadd    za.s[w10, 1, vgx2], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0x5c,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a05c01 <unknown>

fadd    za.s[w8, 5, vgx2], {z22.s, z23.s}  // 11000001-10100000-00011110-11000101
// CHECK-INST: fadd    za.s[w8, 5, vgx2], { z22.s, z23.s }
// CHECK-ENCODING: [0xc5,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01ec5 <unknown>

fadd    za.s[w8, 5], {z22.s, z23.s}  // 11000001-10100000-00011110-11000101
// CHECK-INST: fadd    za.s[w8, 5, vgx2], { z22.s, z23.s }
// CHECK-ENCODING: [0xc5,0x1e,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01ec5 <unknown>

fadd    za.s[w11, 2, vgx2], {z8.s, z9.s}  // 11000001-10100000-01111101-00000010
// CHECK-INST: fadd    za.s[w11, 2, vgx2], { z8.s, z9.s }
// CHECK-ENCODING: [0x02,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d02 <unknown>

fadd    za.s[w11, 2], {z8.s, z9.s}  // 11000001-10100000-01111101-00000010
// CHECK-INST: fadd    za.s[w11, 2, vgx2], { z8.s, z9.s }
// CHECK-ENCODING: [0x02,0x7d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07d02 <unknown>

fadd    za.s[w9, 7, vgx2], {z12.s, z13.s}  // 11000001-10100000-00111101-10000111
// CHECK-INST: fadd    za.s[w9, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0x3d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a03d87 <unknown>

fadd    za.s[w9, 7], {z12.s, z13.s}  // 11000001-10100000-00111101-10000111
// CHECK-INST: fadd    za.s[w9, 7, vgx2], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0x3d,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a03d87 <unknown>


fadd    za.d[w8, 0, vgx4], {z0.d - z3.d}  // 11000001-11100001-00011100-00000000
// CHECK-INST: fadd    za.d[w8, 0, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e11c00 <unknown>

fadd    za.d[w8, 0], {z0.d - z3.d}  // 11000001-11100001-00011100-00000000
// CHECK-INST: fadd    za.d[w8, 0, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e11c00 <unknown>

fadd    za.d[w10, 5, vgx4], {z8.d - z11.d}  // 11000001-11100001-01011101-00000101
// CHECK-INST: fadd    za.d[w10, 5, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x05,0x5d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e15d05 <unknown>

fadd    za.d[w10, 5], {z8.d - z11.d}  // 11000001-11100001-01011101-00000101
// CHECK-INST: fadd    za.d[w10, 5, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x05,0x5d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e15d05 <unknown>

fadd    za.d[w11, 7, vgx4], {z12.d - z15.d}  // 11000001-11100001-01111101-10000111
// CHECK-INST: fadd    za.d[w11, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e17d87 <unknown>

fadd    za.d[w11, 7], {z12.d - z15.d}  // 11000001-11100001-01111101-10000111
// CHECK-INST: fadd    za.d[w11, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e17d87 <unknown>

fadd    za.d[w11, 7, vgx4], {z28.d - z31.d}  // 11000001-11100001-01111111-10000111
// CHECK-INST: fadd    za.d[w11, 7, vgx4], { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0x7f,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e17f87 <unknown>

fadd    za.d[w11, 7], {z28.d - z31.d}  // 11000001-11100001-01111111-10000111
// CHECK-INST: fadd    za.d[w11, 7, vgx4], { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0x7f,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e17f87 <unknown>

fadd    za.d[w8, 5, vgx4], {z16.d - z19.d}  // 11000001-11100001-00011110-00000101
// CHECK-INST: fadd    za.d[w8, 5, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e11e05 <unknown>

fadd    za.d[w8, 5], {z16.d - z19.d}  // 11000001-11100001-00011110-00000101
// CHECK-INST: fadd    za.d[w8, 5, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e11e05 <unknown>

fadd    za.d[w8, 1, vgx4], {z0.d - z3.d}  // 11000001-11100001-00011100-00000001
// CHECK-INST: fadd    za.d[w8, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e11c01 <unknown>

fadd    za.d[w8, 1], {z0.d - z3.d}  // 11000001-11100001-00011100-00000001
// CHECK-INST: fadd    za.d[w8, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x1c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e11c01 <unknown>

fadd    za.d[w10, 0, vgx4], {z16.d - z19.d}  // 11000001-11100001-01011110-00000000
// CHECK-INST: fadd    za.d[w10, 0, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x00,0x5e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e15e00 <unknown>

fadd    za.d[w10, 0], {z16.d - z19.d}  // 11000001-11100001-01011110-00000000
// CHECK-INST: fadd    za.d[w10, 0, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x00,0x5e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e15e00 <unknown>

fadd    za.d[w8, 0, vgx4], {z12.d - z15.d}  // 11000001-11100001-00011101-10000000
// CHECK-INST: fadd    za.d[w8, 0, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x80,0x1d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e11d80 <unknown>

fadd    za.d[w8, 0], {z12.d - z15.d}  // 11000001-11100001-00011101-10000000
// CHECK-INST: fadd    za.d[w8, 0, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x80,0x1d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e11d80 <unknown>

fadd    za.d[w10, 1, vgx4], {z0.d - z3.d}  // 11000001-11100001-01011100-00000001
// CHECK-INST: fadd    za.d[w10, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x5c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e15c01 <unknown>

fadd    za.d[w10, 1], {z0.d - z3.d}  // 11000001-11100001-01011100-00000001
// CHECK-INST: fadd    za.d[w10, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x5c,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e15c01 <unknown>

fadd    za.d[w8, 5, vgx4], {z20.d - z23.d}  // 11000001-11100001-00011110-10000101
// CHECK-INST: fadd    za.d[w8, 5, vgx4], { z20.d - z23.d }
// CHECK-ENCODING: [0x85,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e11e85 <unknown>

fadd    za.d[w8, 5], {z20.d - z23.d}  // 11000001-11100001-00011110-10000101
// CHECK-INST: fadd    za.d[w8, 5, vgx4], { z20.d - z23.d }
// CHECK-ENCODING: [0x85,0x1e,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e11e85 <unknown>

fadd    za.d[w11, 2, vgx4], {z8.d - z11.d}  // 11000001-11100001-01111101-00000010
// CHECK-INST: fadd    za.d[w11, 2, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x02,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e17d02 <unknown>

fadd    za.d[w11, 2], {z8.d - z11.d}  // 11000001-11100001-01111101-00000010
// CHECK-INST: fadd    za.d[w11, 2, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x02,0x7d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e17d02 <unknown>

fadd    za.d[w9, 7, vgx4], {z12.d - z15.d}  // 11000001-11100001-00111101-10000111
// CHECK-INST: fadd    za.d[w9, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x3d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e13d87 <unknown>

fadd    za.d[w9, 7], {z12.d - z15.d}  // 11000001-11100001-00111101-10000111
// CHECK-INST: fadd    za.d[w9, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x3d,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2 sme-f64f64
// CHECK-UNKNOWN: c1e13d87 <unknown>


fadd    za.s[w8, 0, vgx4], {z0.s - z3.s}  // 11000001-10100001-00011100-00000000
// CHECK-INST: fadd    za.s[w8, 0, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c00 <unknown>

fadd    za.s[w8, 0], {z0.s - z3.s}  // 11000001-10100001-00011100-00000000
// CHECK-INST: fadd    za.s[w8, 0, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c00 <unknown>

fadd    za.s[w10, 5, vgx4], {z8.s - z11.s}  // 11000001-10100001-01011101-00000101
// CHECK-INST: fadd    za.s[w10, 5, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x05,0x5d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15d05 <unknown>

fadd    za.s[w10, 5], {z8.s - z11.s}  // 11000001-10100001-01011101-00000101
// CHECK-INST: fadd    za.s[w10, 5, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x05,0x5d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15d05 <unknown>

fadd    za.s[w11, 7, vgx4], {z12.s - z15.s}  // 11000001-10100001-01111101-10000111
// CHECK-INST: fadd    za.s[w11, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x87,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d87 <unknown>

fadd    za.s[w11, 7], {z12.s - z15.s}  // 11000001-10100001-01111101-10000111
// CHECK-INST: fadd    za.s[w11, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x87,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d87 <unknown>

fadd    za.s[w11, 7, vgx4], {z28.s - z31.s}  // 11000001-10100001-01111111-10000111
// CHECK-INST: fadd    za.s[w11, 7, vgx4], { z28.s - z31.s }
// CHECK-ENCODING: [0x87,0x7f,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17f87 <unknown>

fadd    za.s[w11, 7], {z28.s - z31.s}  // 11000001-10100001-01111111-10000111
// CHECK-INST: fadd    za.s[w11, 7, vgx4], { z28.s - z31.s }
// CHECK-ENCODING: [0x87,0x7f,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17f87 <unknown>

fadd    za.s[w8, 5, vgx4], {z16.s - z19.s}  // 11000001-10100001-00011110-00000101
// CHECK-INST: fadd    za.s[w8, 5, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x05,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e05 <unknown>

fadd    za.s[w8, 5], {z16.s - z19.s}  // 11000001-10100001-00011110-00000101
// CHECK-INST: fadd    za.s[w8, 5, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x05,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e05 <unknown>

fadd    za.s[w8, 1, vgx4], {z0.s - z3.s}  // 11000001-10100001-00011100-00000001
// CHECK-INST: fadd    za.s[w8, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c01 <unknown>

fadd    za.s[w8, 1], {z0.s - z3.s}  // 11000001-10100001-00011100-00000001
// CHECK-INST: fadd    za.s[w8, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0x1c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11c01 <unknown>

fadd    za.s[w10, 0, vgx4], {z16.s - z19.s}  // 11000001-10100001-01011110-00000000
// CHECK-INST: fadd    za.s[w10, 0, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x00,0x5e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15e00 <unknown>

fadd    za.s[w10, 0], {z16.s - z19.s}  // 11000001-10100001-01011110-00000000
// CHECK-INST: fadd    za.s[w10, 0, vgx4], { z16.s - z19.s }
// CHECK-ENCODING: [0x00,0x5e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15e00 <unknown>

fadd    za.s[w8, 0, vgx4], {z12.s - z15.s}  // 11000001-10100001-00011101-10000000
// CHECK-INST: fadd    za.s[w8, 0, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x80,0x1d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11d80 <unknown>

fadd    za.s[w8, 0], {z12.s - z15.s}  // 11000001-10100001-00011101-10000000
// CHECK-INST: fadd    za.s[w8, 0, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x80,0x1d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11d80 <unknown>

fadd    za.s[w10, 1, vgx4], {z0.s - z3.s}  // 11000001-10100001-01011100-00000001
// CHECK-INST: fadd    za.s[w10, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0x5c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15c01 <unknown>

fadd    za.s[w10, 1], {z0.s - z3.s}  // 11000001-10100001-01011100-00000001
// CHECK-INST: fadd    za.s[w10, 1, vgx4], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0x5c,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a15c01 <unknown>

fadd    za.s[w8, 5, vgx4], {z20.s - z23.s}  // 11000001-10100001-00011110-10000101
// CHECK-INST: fadd    za.s[w8, 5, vgx4], { z20.s - z23.s }
// CHECK-ENCODING: [0x85,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e85 <unknown>

fadd    za.s[w8, 5], {z20.s - z23.s}  // 11000001-10100001-00011110-10000101
// CHECK-INST: fadd    za.s[w8, 5, vgx4], { z20.s - z23.s }
// CHECK-ENCODING: [0x85,0x1e,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11e85 <unknown>

fadd    za.s[w11, 2, vgx4], {z8.s - z11.s}  // 11000001-10100001-01111101-00000010
// CHECK-INST: fadd    za.s[w11, 2, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x02,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d02 <unknown>

fadd    za.s[w11, 2], {z8.s - z11.s}  // 11000001-10100001-01111101-00000010
// CHECK-INST: fadd    za.s[w11, 2, vgx4], { z8.s - z11.s }
// CHECK-ENCODING: [0x02,0x7d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17d02 <unknown>

fadd    za.s[w9, 7, vgx4], {z12.s - z15.s}  // 11000001-10100001-00111101-10000111
// CHECK-INST: fadd    za.s[w9, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x87,0x3d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a13d87 <unknown>

fadd    za.s[w9, 7], {z12.s - z15.s}  // 11000001-10100001-00111101-10000111
// CHECK-INST: fadd    za.s[w9, 7, vgx4], { z12.s - z15.s }
// CHECK-ENCODING: [0x87,0x3d,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a13d87 <unknown>

