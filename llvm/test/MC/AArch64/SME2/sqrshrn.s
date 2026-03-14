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


sqrshrn z0.b, {z0.s - z3.s}, #32  // 11000001-01100000-11011100-00000000
// CHECK-INST: sqrshrn z0.b, { z0.s - z3.s }, #32
// CHECK-ENCODING: [0x00,0xdc,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c160dc00 <unknown>

sqrshrn z21.b, {z8.s - z11.s}, #11  // 11000001-01110101-11011101-00010101
// CHECK-INST: sqrshrn z21.b, { z8.s - z11.s }, #11
// CHECK-ENCODING: [0x15,0xdd,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c175dd15 <unknown>

sqrshrn z23.b, {z12.s - z15.s}, #24  // 11000001-01101000-11011101-10010111
// CHECK-INST: sqrshrn z23.b, { z12.s - z15.s }, #24
// CHECK-ENCODING: [0x97,0xdd,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c168dd97 <unknown>

sqrshrn z31.b, {z28.s - z31.s}, #1  // 11000001-01111111-11011111-10011111
// CHECK-INST: sqrshrn z31.b, { z28.s - z31.s }, #1
// CHECK-ENCODING: [0x9f,0xdf,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17fdf9f <unknown>


sqrshrn z0.h, {z0.d - z3.d}, #64  // 11000001-10100000-11011100-00000000
// CHECK-INST: sqrshrn z0.h, { z0.d - z3.d }, #64
// CHECK-ENCODING: [0x00,0xdc,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a0dc00 <unknown>

sqrshrn z21.h, {z8.d - z11.d}, #11  // 11000001-11110101-11011101-00010101
// CHECK-INST: sqrshrn z21.h, { z8.d - z11.d }, #11
// CHECK-ENCODING: [0x15,0xdd,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f5dd15 <unknown>

sqrshrn z23.h, {z12.d - z15.d}, #24  // 11000001-11101000-11011101-10010111
// CHECK-INST: sqrshrn z23.h, { z12.d - z15.d }, #24
// CHECK-ENCODING: [0x97,0xdd,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e8dd97 <unknown>

sqrshrn z31.h, {z28.d - z31.d}, #1  // 11000001-11111111-11011111-10011111
// CHECK-INST: sqrshrn z31.h, { z28.d - z31.d }, #1
// CHECK-ENCODING: [0x9f,0xdf,0xff,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ffdf9f <unknown>

