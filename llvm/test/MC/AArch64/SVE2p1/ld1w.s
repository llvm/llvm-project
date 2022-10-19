// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

ld1w    {z0.s-z1.s}, pn8/z, [x0, x0, lsl #2]  // 10100000-00000000-01000000-00000000
// CHECK-INST: ld1w    { z0.s, z1.s }, pn8/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x40,0x00,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0004000 <unknown>

ld1w    {z20.s-z21.s}, pn13/z, [x10, x21, lsl #2]  // 10100000-00010101-01010101-01010100
// CHECK-INST: ld1w    { z20.s, z21.s }, pn13/z, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x54,0x55,0x15,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0155554 <unknown>

ld1w    {z22.s-z23.s}, pn11/z, [x13, x8, lsl #2]  // 10100000-00001000-01001101-10110110
// CHECK-INST: ld1w    { z22.s, z23.s }, pn11/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xb6,0x4d,0x08,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0084db6 <unknown>

ld1w    {z30.s-z31.s}, pn15/z, [sp, xzr, lsl #2]  // 10100000-00011111-01011111-11111110
// CHECK-INST: ld1w    { z30.s, z31.s }, pn15/z, [sp, xzr, lsl #2]
// CHECK-ENCODING: [0xfe,0x5f,0x1f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a01f5ffe <unknown>

ld1w    {z0.s-z1.s}, pn8/z, [x0]  // 10100000-01000000-01000000-00000000
// CHECK-INST: ld1w    { z0.s, z1.s }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0x40,0x40,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0404000 <unknown>

ld1w    {z20.s-z21.s}, pn13/z, [x10, #10, mul vl]  // 10100000-01000101-01010101-01010100
// CHECK-INST: ld1w    { z20.s, z21.s }, pn13/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x54,0x55,0x45,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0455554 <unknown>

ld1w    {z22.s-z23.s}, pn11/z, [x13, #-16, mul vl]  // 10100000-01001000-01001101-10110110
// CHECK-INST: ld1w    { z22.s, z23.s }, pn11/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb6,0x4d,0x48,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0484db6 <unknown>

ld1w    {z30.s-z31.s}, pn15/z, [sp, #-2, mul vl]  // 10100000-01001111-01011111-11111110
// CHECK-INST: ld1w    { z30.s, z31.s }, pn15/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xfe,0x5f,0x4f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a04f5ffe <unknown>

ld1w    {z0.s-z3.s}, pn8/z, [x0, x0, lsl #2]  // 10100000-00000000-11000000-00000000
// CHECK-INST: ld1w    { z0.s - z3.s }, pn8/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0xc0,0x00,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a000c000 <unknown>

ld1w    {z20.s-z23.s}, pn13/z, [x10, x21, lsl #2]  // 10100000-00010101-11010101-01010100
// CHECK-INST: ld1w    { z20.s - z23.s }, pn13/z, [x10, x21, lsl #2]
// CHECK-ENCODING: [0x54,0xd5,0x15,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a015d554 <unknown>

ld1w    {z20.s-z23.s}, pn11/z, [x13, x8, lsl #2]  // 10100000-00001000-11001101-10110100
// CHECK-INST: ld1w    { z20.s - z23.s }, pn11/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xb4,0xcd,0x08,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a008cdb4 <unknown>

ld1w    {z28.s-z31.s}, pn15/z, [sp, xzr, lsl #2]  // 10100000-00011111-11011111-11111100
// CHECK-INST: ld1w    { z28.s - z31.s }, pn15/z, [sp, xzr, lsl #2]
// CHECK-ENCODING: [0xfc,0xdf,0x1f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a01fdffc <unknown>

ld1w    {z0.s-z3.s}, pn8/z, [x0]  // 10100000-01000000-11000000-00000000
// CHECK-INST: ld1w    { z0.s - z3.s }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0xc0,0x40,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a040c000 <unknown>

ld1w    {z20.s-z23.s}, pn13/z, [x10, #20, mul vl]  // 10100000-01000101-11010101-01010100
// CHECK-INST: ld1w    { z20.s - z23.s }, pn13/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x54,0xd5,0x45,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a045d554 <unknown>

ld1w    {z20.s-z23.s}, pn11/z, [x13, #-32, mul vl]  // 10100000-01001000-11001101-10110100
// CHECK-INST: ld1w    { z20.s - z23.s }, pn11/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb4,0xcd,0x48,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a048cdb4 <unknown>

ld1w    {z28.s-z31.s}, pn15/z, [sp, #-4, mul vl]  // 10100000-01001111-11011111-11111100
// CHECK-INST: ld1w    { z28.s - z31.s }, pn15/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfc,0xdf,0x4f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a04fdffc <unknown>
