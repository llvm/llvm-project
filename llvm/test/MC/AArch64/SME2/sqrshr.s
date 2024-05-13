// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sme2 - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=-sme2 - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


sqrshr  z0.h, {z0.s - z1.s}, #16  // 11000001-11100000-11010100-00000000
// CHECK-INST: sqrshr  z0.h, { z0.s, z1.s }, #16
// CHECK-ENCODING: [0x00,0xd4,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e0d400 <unknown>

sqrshr  z21.h, {z10.s - z11.s}, #11  // 11000001-11100101-11010101-01010101
// CHECK-INST: sqrshr  z21.h, { z10.s, z11.s }, #11
// CHECK-ENCODING: [0x55,0xd5,0xe5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e5d555 <unknown>

sqrshr  z23.h, {z12.s - z13.s}, #8  // 11000001-11101000-11010101-10010111
// CHECK-INST: sqrshr  z23.h, { z12.s, z13.s }, #8
// CHECK-ENCODING: [0x97,0xd5,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8d597 <unknown>

sqrshr  z31.h, {z30.s - z31.s}, #1  // 11000001-11101111-11010111-11011111
// CHECK-INST: sqrshr  z31.h, { z30.s, z31.s }, #1
// CHECK-ENCODING: [0xdf,0xd7,0xef,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1efd7df <unknown>


sqrshr  z0.b, {z0.s - z3.s}, #32  // 11000001-01100000-11011000-00000000
// CHECK-INST: sqrshr  z0.b, { z0.s - z3.s }, #32
// CHECK-ENCODING: [0x00,0xd8,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160d800 <unknown>

sqrshr  z21.b, {z8.s - z11.s}, #11  // 11000001-01110101-11011001-00010101
// CHECK-INST: sqrshr  z21.b, { z8.s - z11.s }, #11
// CHECK-ENCODING: [0x15,0xd9,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175d915 <unknown>

sqrshr  z23.b, {z12.s - z15.s}, #24  // 11000001-01101000-11011001-10010111
// CHECK-INST: sqrshr  z23.b, { z12.s - z15.s }, #24
// CHECK-ENCODING: [0x97,0xd9,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168d997 <unknown>

sqrshr  z31.b, {z28.s - z31.s}, #1  // 11000001-01111111-11011011-10011111
// CHECK-INST: sqrshr  z31.b, { z28.s - z31.s }, #1
// CHECK-ENCODING: [0x9f,0xdb,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17fdb9f <unknown>


sqrshr  z0.h, {z0.d - z3.d}, #64  // 11000001-10100000-11011000-00000000
// CHECK-INST: sqrshr  z0.h, { z0.d - z3.d }, #64
// CHECK-ENCODING: [0x00,0xd8,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0d800 <unknown>

sqrshr  z21.h, {z8.d - z11.d}, #11  // 11000001-11110101-11011001-00010101
// CHECK-INST: sqrshr  z21.h, { z8.d - z11.d }, #11
// CHECK-ENCODING: [0x15,0xd9,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5d915 <unknown>

sqrshr  z23.h, {z12.d - z15.d}, #24  // 11000001-11101000-11011001-10010111
// CHECK-INST: sqrshr  z23.h, { z12.d - z15.d }, #24
// CHECK-ENCODING: [0x97,0xd9,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8d997 <unknown>

sqrshr  z31.h, {z28.d - z31.d}, #1  // 11000001-11111111-11011011-10011111
// CHECK-INST: sqrshr  z31.h, { z28.d - z31.d }, #1
// CHECK-ENCODING: [0x9f,0xdb,0xff,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ffdb9f <unknown>

