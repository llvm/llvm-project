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


sel     {z0.h, z1.h}, pn8, {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-01100000-10000000-00000000
// CHECK-INST: sel     { z0.h, z1.h }, pn8, { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x80,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1608000 <unknown>

sel     {z20.h, z21.h}, pn13, {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-01110100-10010101-01010100
// CHECK-INST: sel     { z20.h, z21.h }, pn13, { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x54,0x95,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1749554 <unknown>

sel     {z22.h, z23.h}, pn11, {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-01101000-10001101-10010110
// CHECK-INST: sel     { z22.h, z23.h }, pn11, { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x96,0x8d,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1688d96 <unknown>

sel     {z30.h, z31.h}, pn15, {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-01111110-10011111-11011110
// CHECK-INST: sel     { z30.h, z31.h }, pn15, { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xde,0x9f,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e9fde <unknown>


sel     {z0.s, z1.s}, pn8, {z0.s, z1.s}, {z0.s, z1.s}  // 11000001-10100000-10000000-00000000
// CHECK-INST: sel     { z0.s, z1.s }, pn8, { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x80,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a08000 <unknown>

sel     {z20.s, z21.s}, pn13, {z10.s, z11.s}, {z20.s, z21.s}  // 11000001-10110100-10010101-01010100
// CHECK-INST: sel     { z20.s, z21.s }, pn13, { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x54,0x95,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b49554 <unknown>

sel     {z22.s, z23.s}, pn11, {z12.s, z13.s}, {z8.s, z9.s}  // 11000001-10101000-10001101-10010110
// CHECK-INST: sel     { z22.s, z23.s }, pn11, { z12.s, z13.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x96,0x8d,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a88d96 <unknown>

sel     {z30.s, z31.s}, pn15, {z30.s, z31.s}, {z30.s, z31.s}  // 11000001-10111110-10011111-11011110
// CHECK-INST: sel     { z30.s, z31.s }, pn15, { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xde,0x9f,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be9fde <unknown>


sel     {z0.d, z1.d}, pn8, {z0.d, z1.d}, {z0.d, z1.d}  // 11000001-11100000-10000000-00000000
// CHECK-INST: sel     { z0.d, z1.d }, pn8, { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x80,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e08000 <unknown>

sel     {z20.d, z21.d}, pn13, {z10.d, z11.d}, {z20.d, z21.d}  // 11000001-11110100-10010101-01010100
// CHECK-INST: sel     { z20.d, z21.d }, pn13, { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x54,0x95,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f49554 <unknown>

sel     {z22.d, z23.d}, pn11, {z12.d, z13.d}, {z8.d, z9.d}  // 11000001-11101000-10001101-10010110
// CHECK-INST: sel     { z22.d, z23.d }, pn11, { z12.d, z13.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x96,0x8d,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e88d96 <unknown>

sel     {z30.d, z31.d}, pn15, {z30.d, z31.d}, {z30.d, z31.d}  // 11000001-11111110-10011111-11011110
// CHECK-INST: sel     { z30.d, z31.d }, pn15, { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xde,0x9f,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe9fde <unknown>


sel     {z0.b, z1.b}, pn8, {z0.b, z1.b}, {z0.b, z1.b}  // 11000001-00100000-10000000-00000000
// CHECK-INST: sel     { z0.b, z1.b }, pn8, { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x00,0x80,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1208000 <unknown>

sel     {z20.b, z21.b}, pn13, {z10.b, z11.b}, {z20.b, z21.b}  // 11000001-00110100-10010101-01010100
// CHECK-INST: sel     { z20.b, z21.b }, pn13, { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x54,0x95,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1349554 <unknown>

sel     {z22.b, z23.b}, pn11, {z12.b, z13.b}, {z8.b, z9.b}  // 11000001-00101000-10001101-10010110
// CHECK-INST: sel     { z22.b, z23.b }, pn11, { z12.b, z13.b }, { z8.b, z9.b }
// CHECK-ENCODING: [0x96,0x8d,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1288d96 <unknown>

sel     {z30.b, z31.b}, pn15, {z30.b, z31.b}, {z30.b, z31.b}  // 11000001-00111110-10011111-11011110
// CHECK-INST: sel     { z30.b, z31.b }, pn15, { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xde,0x9f,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e9fde <unknown>


sel     {z0.h - z3.h}, pn8, {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-01100001-10000000-00000000
// CHECK-INST: sel     { z0.h - z3.h }, pn8, { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0x80,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1618000 <unknown>

sel     {z20.h - z23.h}, pn13, {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-01110101-10010101-00010100
// CHECK-INST: sel     { z20.h - z23.h }, pn13, { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x14,0x95,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1759514 <unknown>

sel     {z20.h - z23.h}, pn11, {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-01101001-10001101-10010100
// CHECK-INST: sel     { z20.h - z23.h }, pn11, { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x94,0x8d,0x69,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1698d94 <unknown>

sel     {z28.h - z31.h}, pn15, {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-01111101-10011111-10011100
// CHECK-INST: sel     { z28.h - z31.h }, pn15, { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x9c,0x9f,0x7d,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17d9f9c <unknown>


sel     {z0.s - z3.s}, pn8, {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10100001-10000000-00000000
// CHECK-INST: sel     { z0.s - z3.s }, pn8, { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0x80,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a18000 <unknown>

sel     {z20.s - z23.s}, pn13, {z8.s - z11.s}, {z20.s - z23.s}  // 11000001-10110101-10010101-00010100
// CHECK-INST: sel     { z20.s - z23.s }, pn13, { z8.s - z11.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x14,0x95,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b59514 <unknown>

sel     {z20.s - z23.s}, pn11, {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-10001101-10010100
// CHECK-INST: sel     { z20.s - z23.s }, pn11, { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x94,0x8d,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a98d94 <unknown>

sel     {z28.s - z31.s}, pn15, {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111101-10011111-10011100
// CHECK-INST: sel     { z28.s - z31.s }, pn15, { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x9c,0x9f,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd9f9c <unknown>


sel     {z0.d - z3.d}, pn8, {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-10000000-00000000
// CHECK-INST: sel     { z0.d - z3.d }, pn8, { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x80,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e18000 <unknown>

sel     {z20.d - z23.d}, pn13, {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-10010101-00010100
// CHECK-INST: sel     { z20.d - z23.d }, pn13, { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x14,0x95,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f59514 <unknown>

sel     {z20.d - z23.d}, pn11, {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-10001101-10010100
// CHECK-INST: sel     { z20.d - z23.d }, pn11, { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x94,0x8d,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e98d94 <unknown>

sel     {z28.d - z31.d}, pn15, {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-10011111-10011100
// CHECK-INST: sel     { z28.d - z31.d }, pn15, { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x9c,0x9f,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd9f9c <unknown>


sel     {z0.b - z3.b}, pn8, {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-00100001-10000000-00000000
// CHECK-INST: sel     { z0.b - z3.b }, pn8, { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x00,0x80,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1218000 <unknown>

sel     {z20.b - z23.b}, pn13, {z8.b - z11.b}, {z20.b - z23.b}  // 11000001-00110101-10010101-00010100
// CHECK-INST: sel     { z20.b - z23.b }, pn13, { z8.b - z11.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x14,0x95,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1359514 <unknown>

sel     {z20.b - z23.b}, pn11, {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-00101001-10001101-10010100
// CHECK-INST: sel     { z20.b - z23.b }, pn11, { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x94,0x8d,0x29,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1298d94 <unknown>

sel     {z28.b - z31.b}, pn15, {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-00111101-10011111-10011100
// CHECK-INST: sel     { z28.b - z31.b }, pn15, { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x9c,0x9f,0x3d,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13d9f9c <unknown>

