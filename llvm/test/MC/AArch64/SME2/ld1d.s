// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 --no-print-imm-hex -\
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


ld1d    {z0.d, z8.d}, pn8/z, [x0, x0, lsl #3]  // 10100001-00000000-01100000-00000000
// CHECK-INST: ld1d    { z0.d, z8.d }, pn8/z, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0x60,0x00,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1006000 <unknown>

ld1d    {z21.d, z29.d}, pn13/z, [x10, x21, lsl #3]  // 10100001-00010101-01110101-01010101
// CHECK-INST: ld1d    { z21.d, z29.d }, pn13/z, [x10, x21, lsl #3]
// CHECK-ENCODING: [0x55,0x75,0x15,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1157555 <unknown>

ld1d    {z23.d, z31.d}, pn11/z, [x13, x8, lsl #3]  // 10100001-00001000-01101101-10110111
// CHECK-INST: ld1d    { z23.d, z31.d }, pn11/z, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xb7,0x6d,0x08,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1086db7 <unknown>

ld1d    {z23.d, z31.d}, pn15/z, [sp, xzr, lsl #3]  // 10100001-00011111-01111111-11110111
// CHECK-INST: ld1d    { z23.d, z31.d }, pn15/z, [sp, xzr, lsl #3]
// CHECK-ENCODING: [0xf7,0x7f,0x1f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a11f7ff7 <unknown>


ld1d    {z0.d, z8.d}, pn8/z, [x0]  // 10100001-01000000-01100000-00000000
// CHECK-INST: ld1d    { z0.d, z8.d }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0x60,0x40,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1406000 <unknown>

ld1d    {z21.d, z29.d}, pn13/z, [x10, #10, mul vl]  // 10100001-01000101-01110101-01010101
// CHECK-INST: ld1d    { z21.d, z29.d }, pn13/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0x75,0x45,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1457555 <unknown>

ld1d    {z23.d, z31.d}, pn11/z, [x13, #-16, mul vl]  // 10100001-01001000-01101101-10110111
// CHECK-INST: ld1d    { z23.d, z31.d }, pn11/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0x6d,0x48,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1486db7 <unknown>

ld1d    {z23.d, z31.d}, pn15/z, [sp, #-2, mul vl]  // 10100001-01001111-01111111-11110111
// CHECK-INST: ld1d    { z23.d, z31.d }, pn15/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xf7,0x7f,0x4f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a14f7ff7 <unknown>


ld1d    {z0.d, z4.d, z8.d, z12.d}, pn8/z, [x0, x0, lsl #3]  // 10100001-00000000-11100000-00000000
// CHECK-INST: ld1d    { z0.d, z4.d, z8.d, z12.d }, pn8/z, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0xe0,0x00,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a100e000 <unknown>

ld1d    {z17.d, z21.d, z25.d, z29.d}, pn13/z, [x10, x21, lsl #3]  // 10100001-00010101-11110101-01010001
// CHECK-INST: ld1d    { z17.d, z21.d, z25.d, z29.d }, pn13/z, [x10, x21, lsl #3]
// CHECK-ENCODING: [0x51,0xf5,0x15,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a115f551 <unknown>

ld1d    {z19.d, z23.d, z27.d, z31.d}, pn11/z, [x13, x8, lsl #3]  // 10100001-00001000-11101101-10110011
// CHECK-INST: ld1d    { z19.d, z23.d, z27.d, z31.d }, pn11/z, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xb3,0xed,0x08,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a108edb3 <unknown>

ld1d    {z19.d, z23.d, z27.d, z31.d}, pn15/z, [sp, xzr, lsl #3]  // 10100001-00011111-11111111-11110011
// CHECK-INST: ld1d    { z19.d, z23.d, z27.d, z31.d }, pn15/z, [sp, xzr, lsl #3]
// CHECK-ENCODING: [0xf3,0xff,0x1f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a11ffff3 <unknown>


ld1d    {z0.d, z4.d, z8.d, z12.d}, pn8/z, [x0]  // 10100001-01000000-11100000-00000000
// CHECK-INST: ld1d    { z0.d, z4.d, z8.d, z12.d }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x40,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a140e000 <unknown>

ld1d    {z17.d, z21.d, z25.d, z29.d}, pn13/z, [x10, #20, mul vl]  // 10100001-01000101-11110101-01010001
// CHECK-INST: ld1d    { z17.d, z21.d, z25.d, z29.d }, pn13/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x51,0xf5,0x45,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a145f551 <unknown>

ld1d    {z19.d, z23.d, z27.d, z31.d}, pn11/z, [x13, #-32, mul vl]  // 10100001-01001000-11101101-10110011
// CHECK-INST: ld1d    { z19.d, z23.d, z27.d, z31.d }, pn11/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb3,0xed,0x48,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a148edb3 <unknown>

ld1d    {z19.d, z23.d, z27.d, z31.d}, pn15/z, [sp, #-4, mul vl]  // 10100001-01001111-11111111-11110011
// CHECK-INST: ld1d    { z19.d, z23.d, z27.d, z31.d }, pn15/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xf3,0xff,0x4f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a14ffff3 <unknown>

