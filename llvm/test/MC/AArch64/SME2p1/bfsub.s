// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p1,+b16b16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1,+b16b16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1,+b16b16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfsub   za.h[w8, 0, vgx2], {z0.h, z1.h}  // 11000001-11100100-00011100-00001000
// CHECK-INST: bfsub   za.h[w8, 0, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x08,0x1c,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e41c08 <unknown>

bfsub   za.h[w8, 0], {z0.h - z1.h}  // 11000001-11100100-00011100-00001000
// CHECK-INST: bfsub   za.h[w8, 0, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x08,0x1c,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e41c08 <unknown>

bfsub   za.h[w10, 5, vgx2], {z10.h, z11.h}  // 11000001-11100100-01011101-01001101
// CHECK-INST: bfsub   za.h[w10, 5, vgx2], { z10.h, z11.h }
// CHECK-ENCODING: [0x4d,0x5d,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e45d4d <unknown>

bfsub   za.h[w10, 5], {z10.h - z11.h}  // 11000001-11100100-01011101-01001101
// CHECK-INST: bfsub   za.h[w10, 5, vgx2], { z10.h, z11.h }
// CHECK-ENCODING: [0x4d,0x5d,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e45d4d <unknown>

bfsub   za.h[w11, 7, vgx2], {z12.h, z13.h}  // 11000001-11100100-01111101-10001111
// CHECK-INST: bfsub   za.h[w11, 7, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x8f,0x7d,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e47d8f <unknown>

bfsub   za.h[w11, 7], {z12.h - z13.h}  // 11000001-11100100-01111101-10001111
// CHECK-INST: bfsub   za.h[w11, 7, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x8f,0x7d,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e47d8f <unknown>

bfsub   za.h[w11, 7, vgx2], {z30.h, z31.h}  // 11000001-11100100-01111111-11001111
// CHECK-INST: bfsub   za.h[w11, 7, vgx2], { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x7f,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e47fcf <unknown>

bfsub   za.h[w11, 7], {z30.h - z31.h}  // 11000001-11100100-01111111-11001111
// CHECK-INST: bfsub   za.h[w11, 7, vgx2], { z30.h, z31.h }
// CHECK-ENCODING: [0xcf,0x7f,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e47fcf <unknown>

bfsub   za.h[w8, 5, vgx2], {z16.h, z17.h}  // 11000001-11100100-00011110-00001101
// CHECK-INST: bfsub   za.h[w8, 5, vgx2], { z16.h, z17.h }
// CHECK-ENCODING: [0x0d,0x1e,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e41e0d <unknown>

bfsub   za.h[w8, 5], {z16.h - z17.h}  // 11000001-11100100-00011110-00001101
// CHECK-INST: bfsub   za.h[w8, 5, vgx2], { z16.h, z17.h }
// CHECK-ENCODING: [0x0d,0x1e,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e41e0d <unknown>

bfsub   za.h[w8, 1, vgx2], {z0.h, z1.h}  // 11000001-11100100-00011100-00001001
// CHECK-INST: bfsub   za.h[w8, 1, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x09,0x1c,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e41c09 <unknown>

bfsub   za.h[w8, 1], {z0.h - z1.h}  // 11000001-11100100-00011100-00001001
// CHECK-INST: bfsub   za.h[w8, 1, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x09,0x1c,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e41c09 <unknown>

bfsub   za.h[w10, 0, vgx2], {z18.h, z19.h}  // 11000001-11100100-01011110, 01001000
// CHECK-INST: bfsub   za.h[w10, 0, vgx2], { z18.h, z19.h }
// CHECK-ENCODING: [0x48,0x5e,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e45e48 <unknown>

bfsub   za.h[w10, 0], {z18.h - z19.h}  // 11000001-11100100-01011110-01001000
// CHECK-INST: bfsub   za.h[w10, 0, vgx2], { z18.h, z19.h }
// CHECK-ENCODING: [0x48,0x5e,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e45e48 <unknown>

bfsub   za.h[w8, 0, vgx2], {z12.h, z13.h}  // 11000001-11100100-00011101-10001000
// CHECK-INST: bfsub   za.h[w8, 0, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x88,0x1d,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e41d88 <unknown>

bfsub   za.h[w8, 0], {z12.h - z13.h}  // 11000001-11100100-00011101-10001000
// CHECK-INST: bfsub   za.h[w8, 0, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x88,0x1d,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e41d88 <unknown>

bfsub   za.h[w10, 1, vgx2], {z0.h, z1.h}  // 11000001-11100100-01011100-00001001
// CHECK-INST: bfsub   za.h[w10, 1, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x09,0x5c,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e45c09 <unknown>

bfsub   za.h[w10, 1], {z0.h - z1.h}  // 11000001-11100100-01011100-00001001
// CHECK-INST: bfsub   za.h[w10, 1, vgx2], { z0.h, z1.h }
// CHECK-ENCODING: [0x09,0x5c,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e45c09 <unknown>

bfsub   za.h[w8, 5, vgx2], {z22.h, z23.h}  // 11000001-11100100-00011110, 11001101
// CHECK-INST: bfsub   za.h[w8, 5, vgx2], { z22.h, z23.h }
// CHECK-ENCODING: [0xcd,0x1e,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e41ecd <unknown>

bfsub   za.h[w8, 5], {z22.h - z23.h}  // 11000001-11100100-00011110-11001101
// CHECK-INST: bfsub   za.h[w8, 5, vgx2], { z22.h, z23.h }
// CHECK-ENCODING: [0xcd,0x1e,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e41ecd <unknown>

bfsub   za.h[w11, 2, vgx2], {z8.h, z9.h}  // 11000001-11100100-01111101-00001010
// CHECK-INST: bfsub   za.h[w11, 2, vgx2], { z8.h, z9.h }
// CHECK-ENCODING: [0x0a,0x7d,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e47d0a <unknown>

bfsub   za.h[w11, 2], {z8.h - z9.h}  // 11000001-11100100-01111101-00001010
// CHECK-INST: bfsub   za.h[w11, 2, vgx2], { z8.h, z9.h }
// CHECK-ENCODING: [0x0a,0x7d,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e47d0a <unknown>

bfsub   za.h[w9, 7, vgx2], {z12.h, z13.h}  // 11000001-11100100-00111101-10001111
// CHECK-INST: bfsub   za.h[w9, 7, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x8f,0x3d,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e43d8f <unknown>

bfsub   za.h[w9, 7], {z12.h - z13.h}  // 11000001-11100100-00111101-10001111
// CHECK-INST: bfsub   za.h[w9, 7, vgx2], { z12.h, z13.h }
// CHECK-ENCODING: [0x8f,0x3d,0xe4,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e43d8f <unknown>

bfsub   za.h[w8, 0, vgx4], {z0.h - z3.h}  // 11000001-11100101-00011100-00001000
// CHECK-INST: bfsub   za.h[w8, 0, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x08,0x1c,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e51c08 <unknown>

bfsub   za.h[w8, 0], {z0.h - z3.h}  // 11000001-11100101-00011100-00001000
// CHECK-INST: bfsub   za.h[w8, 0, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x08,0x1c,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e51c08 <unknown>

bfsub   za.h[w10, 5, vgx4], {z8.h - z11.h}  // 11000001-11100101-01011101-00001101
// CHECK-INST: bfsub   za.h[w10, 5, vgx4], { z8.h - z11.h }
// CHECK-ENCODING: [0x0d,0x5d,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e55d0d <unknown>

bfsub   za.h[w10, 5], {z8.h - z11.h}  // 11000001-11100101-01011101-00001101
// CHECK-INST: bfsub   za.h[w10, 5, vgx4], { z8.h - z11.h }
// CHECK-ENCODING: [0x0d,0x5d,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e55d0d <unknown>

bfsub   za.h[w11, 7, vgx4], {z12.h - z15.h}  // 11000001-11100101-01111101-10001111
// CHECK-INST: bfsub   za.h[w11, 7, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x8f,0x7d,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e57d8f <unknown>

bfsub   za.h[w11, 7], {z12.h - z15.h}  // 11000001-11100101-01111101-10001111
// CHECK-INST: bfsub   za.h[w11, 7, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x8f,0x7d,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e57d8f <unknown>

bfsub   za.h[w11, 7, vgx4], {z28.h - z31.h}  // 11000001-11100101-01111111-10001111
// CHECK-INST: bfsub   za.h[w11, 7, vgx4], { z28.h - z31.h }
// CHECK-ENCODING: [0x8f,0x7f,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e57f8f <unknown>

bfsub   za.h[w11, 7], {z28.h - z31.h}  // 11000001-11100101-01111111-10001111
// CHECK-INST: bfsub   za.h[w11, 7, vgx4], { z28.h - z31.h }
// CHECK-ENCODING: [0x8f,0x7f,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e57f8f <unknown>

bfsub   za.h[w8, 5, vgx4], {z16.h - z19.h}  // 11000001-11100101-00011110-00001101
// CHECK-INST: bfsub   za.h[w8, 5, vgx4], { z16.h - z19.h }
// CHECK-ENCODING: [0x0d,0x1e,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e51e0d <unknown>

bfsub   za.h[w8, 5], {z16.h - z19.h}  // 11000001-11100101-00011110-00001101
// CHECK-INST: bfsub   za.h[w8, 5, vgx4], { z16.h - z19.h }
// CHECK-ENCODING: [0x0d,0x1e,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e51e0d <unknown>

bfsub   za.h[w8, 1, vgx4], {z0.h - z3.h}  // 11000001-11100101-00011100-00001001
// CHECK-INST: bfsub   za.h[w8, 1, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x09,0x1c,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e51c09 <unknown>

bfsub   za.h[w8, 1], {z0.h - z3.h}  // 11000001-11100101-00011100-00001001
// CHECK-INST: bfsub   za.h[w8, 1, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x09,0x1c,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e51c09 <unknown>

bfsub   za.h[w10, 0, vgx4], {z16.h - z19.h}  // 11000001-11100101-01011110-00001000
// CHECK-INST: bfsub   za.h[w10, 0, vgx4], { z16.h - z19.h }
// CHECK-ENCODING: [0x08,0x5e,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e55e08 <unknown>

bfsub   za.h[w10, 0], {z16.h - z19.h}  // 11000001-11100101-01011110-00001000
// CHECK-INST: bfsub   za.h[w10, 0, vgx4], { z16.h - z19.h }
// CHECK-ENCODING: [0x08,0x5e,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e55e08 <unknown>

bfsub   za.h[w8, 0, vgx4], {z12.h - z15.h}  // 11000001-11100101-00011101-10001000
// CHECK-INST: bfsub   za.h[w8, 0, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x88,0x1d,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e51d88 <unknown>

bfsub   za.h[w8, 0], {z12.h - z15.h}  // 11000001-11100101-00011101-10001000
// CHECK-INST: bfsub   za.h[w8, 0, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x88,0x1d,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e51d88 <unknown>

bfsub   za.h[w10, 1, vgx4], {z0.h - z3.h}  // 11000001-11100101-01011100-00001001
// CHECK-INST: bfsub   za.h[w10, 1, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x09,0x5c,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e55c09 <unknown>

bfsub   za.h[w10, 1], {z0.h - z3.h}  // 11000001-11100101-01011100-00001001
// CHECK-INST: bfsub   za.h[w10, 1, vgx4], { z0.h - z3.h }
// CHECK-ENCODING: [0x09,0x5c,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e55c09 <unknown>

bfsub   za.h[w8, 5, vgx4], {z20.h - z23.h}  // 11000001-11100101-00011110-10001101
// CHECK-INST: bfsub   za.h[w8, 5, vgx4], { z20.h - z23.h }
// CHECK-ENCODING: [0x8d,0x1e,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e51e8d <unknown>

bfsub   za.h[w8, 5], {z20.h - z23.h}  // 11000001-11100101-00011110-10001101
// CHECK-INST: bfsub   za.h[w8, 5, vgx4], { z20.h - z23.h }
// CHECK-ENCODING: [0x8d,0x1e,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e51e8d <unknown>

bfsub   za.h[w11, 2, vgx4], {z8.h - z11.h}  // 11000001-11100101-01111101-00001010
// CHECK-INST: bfsub   za.h[w11, 2, vgx4], { z8.h - z11.h }
// CHECK-ENCODING: [0x0a,0x7d,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e57d0a <unknown>

bfsub   za.h[w11, 2], {z8.h - z11.h}  // 11000001-11100101-01111101-00001010
// CHECK-INST: bfsub   za.h[w11, 2, vgx4], { z8.h - z11.h }
// CHECK-ENCODING: [0x0a,0x7d,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e57d0a <unknown>

bfsub   za.h[w9, 7, vgx4], {z12.h - z15.h}  // 11000001-11100101-00111101-10001111
// CHECK-INST: bfsub   za.h[w9, 7, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x8f,0x3d,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e53d8f <unknown>

bfsub   za.h[w9, 7], {z12.h - z15.h}  // 11000001-11100101-00111101-10001111
// CHECK-INST: bfsub   za.h[w9, 7, vgx4], { z12.h - z15.h }
// CHECK-ENCODING: [0x8f,0x3d,0xe5,0xc1]
// CHECK-ERROR: instruction requires: b16b16 sme2p1
// CHECK-UNKNOWN: c1e53d8f <unknown>
