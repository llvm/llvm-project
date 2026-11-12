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

ld1d    {z0.d-z1.d}, pn8/z, [x0, x0, lsl #3]  // 10100000-00000000-01100000-00000000
// CHECK-INST: ld1d    { z0.d, z1.d }, pn8/z, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0x60,0x00,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0006000 <unknown>

ld1d    {z20.d-z21.d}, pn13/z, [x10, x21, lsl #3]  // 10100000-00010101-01110101-01010100
// CHECK-INST: ld1d    { z20.d, z21.d }, pn13/z, [x10, x21, lsl #3]
// CHECK-ENCODING: [0x54,0x75,0x15,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0157554 <unknown>

ld1d    {z22.d-z23.d}, pn11/z, [x13, x8, lsl #3]  // 10100000-00001000-01101101-10110110
// CHECK-INST: ld1d    { z22.d, z23.d }, pn11/z, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xb6,0x6d,0x08,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0086db6 <unknown>

ld1d    {z30.d-z31.d}, pn15/z, [sp, xzr, lsl #3]  // 10100000-00011111-01111111-11111110
// CHECK-INST: ld1d    { z30.d, z31.d }, pn15/z, [sp, xzr, lsl #3]
// CHECK-ENCODING: [0xfe,0x7f,0x1f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a01f7ffe <unknown>

ld1d    {z0.d-z1.d}, pn8/z, [x0]  // 10100000-01000000-01100000-00000000
// CHECK-INST: ld1d    { z0.d, z1.d }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0x60,0x40,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0406000 <unknown>

ld1d    {z20.d-z21.d}, pn13/z, [x10, #10, mul vl]  // 10100000-01000101-01110101-01010100
// CHECK-INST: ld1d    { z20.d, z21.d }, pn13/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x54,0x75,0x45,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0457554 <unknown>

ld1d    {z22.d-z23.d}, pn11/z, [x13, #-16, mul vl]  // 10100000-01001000-01101101-10110110
// CHECK-INST: ld1d    { z22.d, z23.d }, pn11/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb6,0x6d,0x48,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0486db6 <unknown>

ld1d    {z30.d-z31.d}, pn15/z, [sp, #-2, mul vl]  // 10100000-01001111-01111111-11111110
// CHECK-INST: ld1d    { z30.d, z31.d }, pn15/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xfe,0x7f,0x4f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a04f7ffe <unknown>

ld1d    {z0.d-z3.d}, pn8/z, [x0, x0, lsl #3]  // 10100000-00000000-11100000-00000000
// CHECK-INST: ld1d    { z0.d - z3.d }, pn8/z, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0xe0,0x00,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a000e000 <unknown>

ld1d    {z20.d-z23.d}, pn13/z, [x10, x21, lsl #3]  // 10100000-00010101-11110101-01010100
// CHECK-INST: ld1d    { z20.d - z23.d }, pn13/z, [x10, x21, lsl #3]
// CHECK-ENCODING: [0x54,0xf5,0x15,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a015f554 <unknown>

ld1d    {z20.d-z23.d}, pn11/z, [x13, x8, lsl #3]  // 10100000-00001000-11101101-10110100
// CHECK-INST: ld1d    { z20.d - z23.d }, pn11/z, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xb4,0xed,0x08,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a008edb4 <unknown>

ld1d    {z28.d-z31.d}, pn15/z, [sp, xzr, lsl #3]  // 10100000-00011111-11111111-11111100
// CHECK-INST: ld1d    { z28.d - z31.d }, pn15/z, [sp, xzr, lsl #3]
// CHECK-ENCODING: [0xfc,0xff,0x1f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a01ffffc <unknown>

ld1d    {z0.d-z3.d}, pn8/z, [x0]  // 10100000-01000000-11100000-00000000
// CHECK-INST: ld1d    { z0.d - z3.d }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x40,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a040e000 <unknown>

ld1d    {z20.d-z23.d}, pn13/z, [x10, #20, mul vl]  // 10100000-01000101-11110101-01010100
// CHECK-INST: ld1d    { z20.d - z23.d }, pn13/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x54,0xf5,0x45,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a045f554 <unknown>

ld1d    {z20.d-z23.d}, pn11/z, [x13, #-32, mul vl]  // 10100000-01001000-11101101-10110100
// CHECK-INST: ld1d    { z20.d - z23.d }, pn11/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb4,0xed,0x48,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a048edb4 <unknown>

ld1d    {z28.d-z31.d}, pn15/z, [sp, #-4, mul vl]  // 10100000-01001111-11111111-11111100
// CHECK-INST: ld1d    { z28.d - z31.d }, pn15/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfc,0xff,0x4f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a04ffffc <unknown>

