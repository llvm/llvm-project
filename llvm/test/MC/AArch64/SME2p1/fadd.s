// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p1,+sme-f16f16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+sme-f16f16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1,+sme-f16f16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fadd    za.h[w8, 0, vgx2], {z0.h, z1.h}  // 11000001-10100100-00011100-00000000
// CHECK-INST: fadd    za.h[w8, 0, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x1c,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a41c00 <unknown>

fadd    za.h[w8, 0], {z0.h - z1.h}  // 11000001-10100100-00011100-00000000
// CHECK-INST: fadd    za.h[w8, 0, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x1c,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a41c00 <unknown>

fadd    za.h[w10, 5, vgx2], {z10.h, z11.h}  // 11000001-10100100-01011101-01000101
// CHECK-INST: fadd    za.h[w10, 5, vgx2], { z10.h, z11.h }
// CHECK-ENCODING: [0x45,0x5d,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a45d45 <unknown>

fadd    za.h[w10, 5], {z10.h - z11.h}  // 11000001-10100100-01011101-01000101
// CHECK-INST: fadd    za.h[w10, 5, vgx2], { z10.h, z11.h }
// CHECK-ENCODING: [0x45,0x5d,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a45d45 <unknown>

fadd    za.h[w11, 7, vgx2], {z12.h, z13.h}  // 11000001-10100100-01111101-10000111
// CHECK-INST: fadd    za.h[w11, 7, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0x7d,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a47d87 <unknown>

fadd    za.h[w11, 7], {z12.h - z13.h}  // 11000001-10100100-01111101-10000111
// CHECK-INST: fadd    za.h[w11, 7, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0x7d,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a47d87 <unknown>

fadd    za.h[w11, 7, vgx2], {z30.h, z31.h}  // 11000001-10100100-01111111-11000111
// CHECK-INST: fadd    za.h[w11, 7, vgx2], { z30.h, z31.h }
// CHECK-ENCODING: [0xc7,0x7f,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a47fc7 <unknown>

fadd    za.h[w11, 7], {z30.h - z31.h}  // 11000001-10100100-01111111-11000111
// CHECK-INST: fadd    za.h[w11, 7, vgx2], { z30.h, z31.h }
// CHECK-ENCODING: [0xc7,0x7f,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a47fc7 <unknown>

fadd    za.h[w8, 5, vgx2], {z16.h, z17.h}  // 11000001-10100100-00011110-00000101
// CHECK-INST: fadd    za.h[w8, 5, vgx2], { z16.h, z17.h }
// CHECK-ENCODING: [0x05,0x1e,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a41e05 <unknown>

fadd    za.h[w8, 5], {z16.h - z17.h}  // 11000001-10100100-00011110-00000101
// CHECK-INST: fadd    za.h[w8, 5, vgx2], { z16.h, z17.h }
// CHECK-ENCODING: [0x05,0x1e,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a41e05 <unknown>

fadd    za.h[w8, 1, vgx2], {z0.h, z1.h}  // 11000001-10100100-00011100-00000001
// CHECK-INST: fadd    za.h[w8, 1, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0x1c,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a41c01 <unknown>

fadd    za.h[w8, 1], {z0.h - z1.h}  // 11000001-10100100-00011100-00000001
// CHECK-INST: fadd    za.h[w8, 1, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0x1c,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a41c01 <unknown>

fadd    za.h[w10, 0, vgx2], {z18.h, z19.h}  // 11000001-10100100-01011110, 01000000
// CHECK-INST: fadd    za.h[w10, 0, vgx2], { z18.h, z19.h }
// CHECK-ENCODING: [0x40,0x5e,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a45e40 <unknown>

fadd    za.h[w10, 0], {z18.h - z19.h}  // 11000001-10100100-01011110-01000000
// CHECK-INST: fadd    za.h[w10, 0, vgx2], { z18.h, z19.h }
// CHECK-ENCODING: [0x40,0x5e,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a45e40 <unknown>

fadd    za.h[w8, 0, vgx2], {z12.h, z13.h}  // 11000001-10100100-00011101-10000000
// CHECK-INST: fadd    za.h[w8, 0, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x80,0x1d,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a41d80 <unknown>

fadd    za.h[w8, 0], {z12.h - z13.h}  // 11000001-10100100-00011101-10000000
// CHECK-INST: fadd    za.h[w8, 0, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x80,0x1d,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a41d80 <unknown>

fadd    za.h[w10, 1, vgx2], {z0.h, z1.h}  // 11000001-10100100-01011100-00000001
// CHECK-INST: fadd    za.h[w10, 1, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0x5c,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a45c01 <unknown>

fadd    za.h[w10, 1], {z0.h - z1.h}  // 11000001-10100100-01011100-00000001
// CHECK-INST: fadd    za.h[w10, 1, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0x5c,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a45c01 <unknown>

fadd    za.h[w8, 5, vgx2], {z22.h, z23.h}  // 11000001-10100100-00011110, 11000101
// CHECK-INST: fadd    za.h[w8, 5, vgx2], { z22.h, z23.h }
// CHECK-ENCODING: [0xc5,0x1e,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a41ec5 <unknown>

fadd    za.h[w8, 5], {z22.h - z23.h}  // 11000001-10100100-00011110-11000101
// CHECK-INST: fadd    za.h[w8, 5, vgx2], { z22.h, z23.h }
// CHECK-ENCODING: [0xc5,0x1e,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a41ec5 <unknown>

fadd    za.h[w11, 2, vgx2], {z8.h, z9.h}  // 11000001-10100100-01111101-00000010
// CHECK-INST: fadd    za.h[w11, 2, vgx2], { z8.h, z9.h }
// CHECK-ENCODING: [0x02,0x7d,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a47d02 <unknown>

fadd    za.h[w11, 2], {z8.h - z9.h}  // 11000001-10100100-01111101-00000010
// CHECK-INST: fadd    za.h[w11, 2, vgx2], { z8.h, z9.h }
// CHECK-ENCODING: [0x02,0x7d,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a47d02 <unknown>

fadd    za.h[w9, 7, vgx2], {z12.h, z13.h}  // 11000001-10100100-00111101-10000111
// CHECK-INST: fadd    za.h[w9, 7, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0x3d,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a43d87 <unknown>

fadd    za.h[w9, 7], {z12.h - z13.h}  // 11000001-10100100-00111101-10000111
// CHECK-INST: fadd    za.h[w9, 7, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0x3d,0xa4,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a43d87 <unknown>

fadd    za.h[w8, 0, vgx4], {z0.h - z3.h}  // 11000001-10100101-00011100-00000000
// CHECK-INST: fadd    za.h[w8, 0, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0x1c,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a51c00 <unknown>

fadd    za.h[w8, 0], {z0.h - z3.h}  // 11000001-10100101-00011100-00000000
// CHECK-INST: fadd    za.h[w8, 0, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0x1c,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a51c00 <unknown>

fadd    za.h[w10, 5, vgx4], {z8.h - z11.h}  // 11000001-10100101-01011101-00000101
// CHECK-INST: fadd    za.h[w10, 5, vgx4], { z8.h - z11.h }
// CHECK-ENCODING: [0x05,0x5d,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a55d05 <unknown>

fadd    za.h[w10, 5], {z8.h - z11.h}  // 11000001-10100101-01011101-00000101
// CHECK-INST: fadd    za.h[w10, 5, vgx4], { z8.h - z11.h }
// CHECK-ENCODING: [0x05,0x5d,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a55d05 <unknown>

fadd    za.h[w11, 7, vgx4], {z12.h - z15.h}  // 11000001-10100101-01111101-10000111
// CHECK-INST: fadd    za.h[w11, 7, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x87,0x7d,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a57d87 <unknown>

fadd    za.h[w11, 7], {z12.h - z15.h}  // 11000001-10100101-01111101-10000111
// CHECK-INST: fadd    za.h[w11, 7, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x87,0x7d,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a57d87 <unknown>

fadd    za.h[w11, 7, vgx4], {z28.h - z31.h}  // 11000001-10100101-01111111-10000111
// CHECK-INST: fadd    za.h[w11, 7, vgx4], { z28.h - z31.h }
// CHECK-ENCODING: [0x87,0x7f,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a57f87 <unknown>

fadd    za.h[w11, 7], {z28.h - z31.h}  // 11000001-10100101-01111111-10000111
// CHECK-INST: fadd    za.h[w11, 7, vgx4], { z28.h - z31.h }
// CHECK-ENCODING: [0x87,0x7f,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a57f87 <unknown>

fadd    za.h[w8, 5, vgx4], {z16.h - z19.h}  // 11000001-10100101-00011110-00000101
// CHECK-INST: fadd    za.h[w8, 5, vgx4], { z16.h - z19.h }
// CHECK-ENCODING: [0x05,0x1e,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a51e05 <unknown>

fadd    za.h[w8, 5], {z16.h - z19.h}  // 11000001-10100101-00011110-00000101
// CHECK-INST: fadd    za.h[w8, 5, vgx4], { z16.h - z19.h }
// CHECK-ENCODING: [0x05,0x1e,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a51e05 <unknown>

fadd    za.h[w8, 1, vgx4], {z0.h - z3.h}  // 11000001-10100101-00011100-00000001
// CHECK-INST: fadd    za.h[w8, 1, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0x1c,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a51c01 <unknown>

fadd    za.h[w8, 1], {z0.h - z3.h}  // 11000001-10100101-00011100-00000001
// CHECK-INST: fadd    za.h[w8, 1, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0x1c,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a51c01 <unknown>

fadd    za.h[w10, 0, vgx4], {z16.h - z19.h}  // 11000001-10100101-01011110-00000000
// CHECK-INST: fadd    za.h[w10, 0, vgx4], { z16.h - z19.h }
// CHECK-ENCODING: [0x00,0x5e,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a55e00 <unknown>

fadd    za.h[w10, 0], {z16.h - z19.h}  // 11000001-10100101-01011110-00000000
// CHECK-INST: fadd    za.h[w10, 0, vgx4], { z16.h - z19.h }
// CHECK-ENCODING: [0x00,0x5e,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a55e00 <unknown>

fadd    za.h[w8, 0, vgx4], {z12.h - z15.h}  // 11000001-10100101-00011101-10000000
// CHECK-INST: fadd    za.h[w8, 0, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x80,0x1d,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a51d80 <unknown>

fadd    za.h[w8, 0], {z12.h - z15.h}  // 11000001-10100101-00011101-10000000
// CHECK-INST: fadd    za.h[w8, 0, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x80,0x1d,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a51d80 <unknown>

fadd    za.h[w10, 1, vgx4], {z0.h - z3.h}  // 11000001-10100101-01011100-00000001
// CHECK-INST: fadd    za.h[w10, 1, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0x5c,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a55c01 <unknown>

fadd    za.h[w10, 1], {z0.h - z3.h}  // 11000001-10100101-01011100-00000001
// CHECK-INST: fadd    za.h[w10, 1, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0x5c,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a55c01 <unknown>

fadd    za.h[w8, 5, vgx4], {z20.h - z23.h}  // 11000001-10100101-00011110-10000101
// CHECK-INST: fadd    za.h[w8, 5, vgx4], { z20.h - z23.h }
// CHECK-ENCODING: [0x85,0x1e,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a51e85 <unknown>

fadd    za.h[w8, 5], {z20.h - z23.h}  // 11000001-10100101-00011110-10000101
// CHECK-INST: fadd    za.h[w8, 5, vgx4], { z20.h - z23.h }
// CHECK-ENCODING: [0x85,0x1e,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a51e85 <unknown>

fadd    za.h[w11, 2, vgx4], {z8.h - z11.h}  // 11000001-10100101-01111101-00000010
// CHECK-INST: fadd    za.h[w11, 2, vgx4], { z8.h - z11.h }
// CHECK-ENCODING: [0x02,0x7d,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a57d02 <unknown>

fadd    za.h[w11, 2], {z8.h - z11.h}  // 11000001-10100101-01111101-00000010
// CHECK-INST: fadd    za.h[w11, 2, vgx4], { z8.h - z11.h }
// CHECK-ENCODING: [0x02,0x7d,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a57d02 <unknown>

fadd    za.h[w9, 7, vgx4], {z12.h - z15.h}  // 11000001-10100101-00111101-10000111
// CHECK-INST: fadd    za.h[w9, 7, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x87,0x3d,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a53d87 <unknown>

fadd    za.h[w9, 7], {z12.h - z15.h}  // 11000001-10100101-00111101-10000111
// CHECK-INST: fadd    za.h[w9, 7, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x87,0x3d,0xa5,0xc1]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c1a53d87 <unknown>
