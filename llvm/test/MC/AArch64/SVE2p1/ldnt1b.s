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

ldnt1b  {z0.b-z1.b}, pn8/z, [x0, x0]  // 10100000-00000000-00000000-00000001
// CHECK-INST: ldnt1b  { z0.b, z1.b }, pn8/z, [x0, x0]
// CHECK-ENCODING: [0x01,0x00,0x00,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0000001 <unknown>

ldnt1b  {z20.b-z21.b}, pn13/z, [x10, x21]  // 10100000-00010101-00010101-01010101
// CHECK-INST: ldnt1b  { z20.b, z21.b }, pn13/z, [x10, x21]
// CHECK-ENCODING: [0x55,0x15,0x15,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0151555 <unknown>

ldnt1b  {z22.b-z23.b}, pn11/z, [x13, x8]  // 10100000-00001000-00001101-10110111
// CHECK-INST: ldnt1b  { z22.b, z23.b }, pn11/z, [x13, x8]
// CHECK-ENCODING: [0xb7,0x0d,0x08,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0080db7 <unknown>

ldnt1b  {z30.b-z31.b}, pn15/z, [sp, xzr]  // 10100000-00011111-00011111-11111111
// CHECK-INST: ldnt1b  { z30.b, z31.b }, pn15/z, [sp, xzr]
// CHECK-ENCODING: [0xff,0x1f,0x1f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a01f1fff <unknown>

ldnt1b  {z0.b-z1.b}, pn8/z, [x0]  // 10100000-01000000-00000000-00000001
// CHECK-INST: ldnt1b  { z0.b, z1.b }, pn8/z, [x0]
// CHECK-ENCODING: [0x01,0x00,0x40,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0400001 <unknown>

ldnt1b  {z20.b-z21.b}, pn13/z, [x10, #10, mul vl]  // 10100000-01000101-00010101-01010101
// CHECK-INST: ldnt1b  { z20.b, z21.b }, pn13/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0x15,0x45,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0451555 <unknown>

ldnt1b  {z22.b-z23.b}, pn11/z, [x13, #-16, mul vl]  // 10100000-01001000-00001101-10110111
// CHECK-INST: ldnt1b  { z22.b, z23.b }, pn11/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0x0d,0x48,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0480db7 <unknown>

ldnt1b  {z30.b-z31.b}, pn15/z, [sp, #-2, mul vl]  // 10100000-01001111-00011111-11111111
// CHECK-INST: ldnt1b  { z30.b, z31.b }, pn15/z, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xff,0x1f,0x4f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a04f1fff <unknown>

ldnt1b  {z0.b-z3.b}, pn8/z, [x0, x0]  // 10100000-00000000-10000000-00000001
// CHECK-INST: ldnt1b  { z0.b - z3.b }, pn8/z, [x0, x0]
// CHECK-ENCODING: [0x01,0x80,0x00,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0008001 <unknown>

ldnt1b  {z20.b-z23.b}, pn13/z, [x10, x21]  // 10100000-00010101-10010101-01010101
// CHECK-INST: ldnt1b  { z20.b - z23.b }, pn13/z, [x10, x21]
// CHECK-ENCODING: [0x55,0x95,0x15,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0159555 <unknown>

ldnt1b  {z20.b-z23.b}, pn11/z, [x13, x8]  // 10100000-00001000-10001101-10110101
// CHECK-INST: ldnt1b  { z20.b - z23.b }, pn11/z, [x13, x8]
// CHECK-ENCODING: [0xb5,0x8d,0x08,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0088db5 <unknown>

ldnt1b  {z28.b-z31.b}, pn15/z, [sp, xzr]  // 10100000-00011111-10011111-11111101
// CHECK-INST: ldnt1b  { z28.b - z31.b }, pn15/z, [sp, xzr]
// CHECK-ENCODING: [0xfd,0x9f,0x1f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a01f9ffd <unknown>

ldnt1b  {z0.b-z3.b}, pn8/z, [x0]  // 10100000-01000000-10000000-00000001
// CHECK-INST: ldnt1b  { z0.b - z3.b }, pn8/z, [x0]
// CHECK-ENCODING: [0x01,0x80,0x40,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0408001 <unknown>

ldnt1b  {z20.b-z23.b}, pn13/z, [x10, #20, mul vl]  // 10100000-01000101-10010101-01010101
// CHECK-INST: ldnt1b  { z20.b - z23.b }, pn13/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0x95,0x45,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0459555 <unknown>

ldnt1b  {z20.b-z23.b}, pn11/z, [x13, #-32, mul vl]  // 10100000-01001000-10001101-10110101
// CHECK-INST: ldnt1b  { z20.b - z23.b }, pn11/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb5,0x8d,0x48,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0488db5 <unknown>

ldnt1b  {z28.b-z31.b}, pn15/z, [sp, #-4, mul vl]  // 10100000-01001111-10011111-11111101
// CHECK-INST: ldnt1b  { z28.b - z31.b }, pn15/z, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfd,0x9f,0x4f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a04f9ffd <unknown>
