// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 --no-print-imm-hex  - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


ld1b    {z0.b, z8.b}, pn8/z, [x0, x0]  // 10100001-00000000-00000000-00000000
// CHECK-INST: ld1b    { z0.b, z8.b }, pn8/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x00,0x00,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1000000 <unknown>

ld1b    {z21.b, z29.b}, pn13/z, [x10, x21]  // 10100001-00010101-00010101-01010101
// CHECK-INST: ld1b    { z21.b, z29.b }, pn13/z, [x10, x21]
// CHECK-ENCODING: [0x55,0x15,0x15,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1151555 <unknown>

ld1b    {z23.b, z31.b}, pn11/z, [x13, x8]  // 10100001-00001000-00001101-10110111
// CHECK-INST: ld1b    { z23.b, z31.b }, pn11/z, [x13, x8]
// CHECK-ENCODING: [0xb7,0x0d,0x08,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1080db7 <unknown>

ld1b    {z23.b, z31.b}, pn15/z, [sp, xzr]  // 10100001-00011111-00011111-11110111
// CHECK-INST: ld1b    { z23.b, z31.b }, pn15/z, [sp, xzr]
// CHECK-ENCODING: [0xf7,0x1f,0x1f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a11f1ff7 <unknown>


ld1b    {z0.b, z8.b}, pn8/z, [x0]  // 10100001-01000000-00000000-00000000
// CHECK-INST: ld1b    { z0.b, z8.b }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0x00,0x40,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1400000 <unknown>

ld1b    {z21.b, z29.b}, pn13/z, [x10, #10, mul vl]  // 10100001-01000101-00010101-01010101
// CHECK-INST: ld1b    { z21.b, z29.b }, pn13/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0x15,0x45,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1451555 <unknown>

ld1b    {z23.b, z31.b}, pn11/z, [x13, #-16, mul vl]  // 10100001-01001000-00001101-10110111
// CHECK-INST: ld1b    { z23.b, z31.b }, pn11/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0x0d,0x48,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1480db7 <unknown>

ld1b    {z23.b, z31.b}, pn15/z, [sp, #-2, mul vl]  // 10100001-01001111-00011111-11110111
// CHECK-INST: ld1b    { z23.b, z31.b }, pn15/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xf7,0x1f,0x4f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a14f1ff7 <unknown>


ld1b    {z0.b, z4.b, z8.b, z12.b}, pn8/z, [x0, x0]  // 10100001-00000000-10000000-00000000
// CHECK-INST: ld1b    { z0.b, z4.b, z8.b, z12.b }, pn8/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x80,0x00,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1008000 <unknown>

ld1b    {z17.b, z21.b, z25.b, z29.b}, pn13/z, [x10, x21]  // 10100001-00010101-10010101-01010001
// CHECK-INST: ld1b    { z17.b, z21.b, z25.b, z29.b }, pn13/z, [x10, x21]
// CHECK-ENCODING: [0x51,0x95,0x15,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1159551 <unknown>

ld1b    {z19.b, z23.b, z27.b, z31.b}, pn11/z, [x13, x8]  // 10100001-00001000-10001101-10110011
// CHECK-INST: ld1b    { z19.b, z23.b, z27.b, z31.b }, pn11/z, [x13, x8]
// CHECK-ENCODING: [0xb3,0x8d,0x08,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1088db3 <unknown>

ld1b    {z19.b, z23.b, z27.b, z31.b}, pn15/z, [sp, xzr]  // 10100001-00011111-10011111-11110011
// CHECK-INST: ld1b    { z19.b, z23.b, z27.b, z31.b }, pn15/z, [sp, xzr]
// CHECK-ENCODING: [0xf3,0x9f,0x1f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a11f9ff3 <unknown>


ld1b    {z0.b, z4.b, z8.b, z12.b}, pn8/z, [x0]  // 10100001-01000000-10000000-00000000
// CHECK-INST: ld1b    { z0.b, z4.b, z8.b, z12.b }, pn8/z, [x0]
// CHECK-ENCODING: [0x00,0x80,0x40,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1408000 <unknown>

ld1b    {z17.b, z21.b, z25.b, z29.b}, pn13/z, [x10, #20, mul vl]  // 10100001-01000101-10010101-01010001
// CHECK-INST: ld1b    { z17.b, z21.b, z25.b, z29.b }, pn13/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x51,0x95,0x45,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1459551 <unknown>

ld1b    {z19.b, z23.b, z27.b, z31.b}, pn11/z, [x13, #-32, mul vl]  // 10100001-01001000-10001101-10110011
// CHECK-INST: ld1b    { z19.b, z23.b, z27.b, z31.b }, pn11/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb3,0x8d,0x48,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a1488db3 <unknown>

ld1b    {z19.b, z23.b, z27.b, z31.b}, pn15/z, [sp, #-4, mul vl]  // 10100001-01001111-10011111-11110011
// CHECK-INST: ld1b    { z19.b, z23.b, z27.b, z31.b }, pn15/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xf3,0x9f,0x4f,0xa1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: a14f9ff3 <unknown>

