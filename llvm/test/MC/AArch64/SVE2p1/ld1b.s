// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

ld1b    {z0.b-z1.b}, pn8/z, [x0, x0]  // 10100000-00000000-00000000-00000000
// CHECK-INST: ld1b    { z0.b, z1.b }, pn8/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x00,0x00,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0000000 <unknown>

ld1b    {z20.b-z21.b}, pn13/z, [x10, x21]  // 10100000-00010101-00010101-01010100
// CHECK-INST: ld1b    { z20.b, z21.b }, pn13/z, [x10, x21]
// CHECK-ENCODING: [0x54,0x15,0x15,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0151554 <unknown>

ld1b    {z22.b-z23.b}, pn11/z, [x13, x8]  // 10100000-00001000-00001101-10110110
// CHECK-INST: ld1b    { z22.b, z23.b }, pn11/z, [x13, x8]
// CHECK-ENCODING: [0xb6,0x0d,0x08,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0080db6 <unknown>

ld1b    {z30.b-z31.b}, pn15/z, [sp, xzr]  // 10100000-00011111-00011111-11111110
// CHECK-INST: ld1b    { z30.b, z31.b }, pn15/z, [sp, xzr]
// CHECK-ENCODING: [0xfe,0x1f,0x1f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a01f1ffe <unknown>

ld1b    {z0.b-z1.b}, pn8/z, [x0]  // 10100000-01000000-00000000-00000000
// CHECK-INST: ld1b    { z0.b, z1.b }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0x00,0x40,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0400000 <unknown>

ld1b    {z20.b-z21.b}, pn13/z, [x10, #10, mul vl]  // 10100000-01000101-00010101-01010100
// CHECK-INST: ld1b    { z20.b, z21.b }, pn13/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x54,0x15,0x45,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0451554 <unknown>

ld1b    {z22.b-z23.b}, pn11/z, [x13, #-16, mul vl]  // 10100000-01001000-00001101-10110110
// CHECK-INST: ld1b    { z22.b, z23.b }, pn11/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb6,0x0d,0x48,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0480db6 <unknown>

ld1b    {z30.b-z31.b}, pn15/z, [sp, #-2, mul vl]  // 10100000-01001111-00011111-11111110
// CHECK-INST: ld1b    { z30.b, z31.b }, pn15/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xfe,0x1f,0x4f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a04f1ffe <unknown>

ld1b    {z0.b-z3.b}, pn8/z, [x0, x0]  // 10100000-00000000-10000000-00000000
// CHECK-INST: ld1b    { z0.b - z3.b }, pn8/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x80,0x00,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0008000 <unknown>

ld1b    {z20.b-z23.b}, pn13/z, [x10, x21]  // 10100000-00010101-10010101-01010100
// CHECK-INST: ld1b    { z20.b - z23.b }, pn13/z, [x10, x21]
// CHECK-ENCODING: [0x54,0x95,0x15,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0159554 <unknown>

ld1b    {z20.b-z23.b}, pn11/z, [x13, x8]  // 10100000-00001000-10001101-10110100
// CHECK-INST: ld1b    { z20.b - z23.b }, pn11/z, [x13, x8]
// CHECK-ENCODING: [0xb4,0x8d,0x08,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0088db4 <unknown>

ld1b    {z28.b-z31.b}, pn15/z, [sp, xzr]  // 10100000-00011111-10011111-11111100
// CHECK-INST: ld1b    { z28.b - z31.b }, pn15/z, [sp, xzr]
// CHECK-ENCODING: [0xfc,0x9f,0x1f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a01f9ffc <unknown>

ld1b    {z0.b-z3.b}, pn8/z, [x0]  // 10100000-01000000-10000000-00000000
// CHECK-INST: ld1b    { z0.b - z3.b }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0x80,0x40,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0408000 <unknown>

ld1b    {z20.b-z23.b}, pn13/z, [x10, #20, mul vl]  // 10100000-01000101-10010101-01010100
// CHECK-INST: ld1b    { z20.b - z23.b }, pn13/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x54,0x95,0x45,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0459554 <unknown>

ld1b    {z20.b-z23.b}, pn11/z, [x13, #-32, mul vl]  // 10100000-01001000-10001101-10110100
// CHECK-INST: ld1b    { z20.b - z23.b }, pn11/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb4,0x8d,0x48,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0488db4 <unknown>

ld1b    {z28.b-z31.b}, pn15/z, [sp, #-4, mul vl]  // 10100000-01001111-10011111-11111100
// CHECK-INST: ld1b    { z28.b - z31.b }, pn15/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfc,0x9f,0x4f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a04f9ffc <unknown>
