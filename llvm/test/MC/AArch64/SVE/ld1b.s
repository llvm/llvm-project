// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld1b     z0.b, p0/z, [x0]
// CHECK-INST: ld1b     { z0.b }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x00,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 00 a4 <unknown>

ld1b     z0.h, p0/z, [x0]
// CHECK-INST: ld1b     { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x20,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 20 a4 <unknown>

ld1b     z0.s, p0/z, [x0]
// CHECK-INST: ld1b     { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x40,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 40 a4 <unknown>

ld1b     z0.d, p0/z, [x0]
// CHECK-INST: ld1b     { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x60,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 60 a4 <unknown>

ld1b    { z0.b }, p0/z, [x0]
// CHECK-INST: ld1b    { z0.b }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x00,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 00 a4 <unknown>

ld1b    { z0.h }, p0/z, [x0]
// CHECK-INST: ld1b    { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x20,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 20 a4 <unknown>

ld1b    { z0.s }, p0/z, [x0]
// CHECK-INST: ld1b    { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x40,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 40 a4 <unknown>

ld1b    { z0.d }, p0/z, [x0]
// CHECK-INST: ld1b    { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x60,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 60 a4 <unknown>

ld1b    { z31.b }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1b    { z31.b }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x0f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 0f a4 <unknown>

ld1b    { z21.b }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1b    { z21.b }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x05,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 05 a4 <unknown>

ld1b    { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1b    { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x2f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 2f a4 <unknown>

ld1b    { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1b    { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x25,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 25 a4 <unknown>

ld1b    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1b    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x4f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 4f a4 <unknown>

ld1b    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1b    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x45,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 45 a4 <unknown>

ld1b    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1b    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x6f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 6f a4 <unknown>

ld1b    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1b    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x65,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 65 a4 <unknown>

ld1b    { z0.b }, p0/z, [sp, x0]
// CHECK-INST: ld1b    { z0.b }, p0/z, [sp, x0]
// CHECK-ENCODING: [0xe0,0x43,0x00,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 43 00 a4 <unknown>

ld1b    { z0.b }, p0/z, [x0, x0]
// CHECK-INST: ld1b    { z0.b }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x40,0x00,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 00 a4 <unknown>

ld1b    { z0.b }, p0/z, [x0, x0, lsl #0]
// CHECK-INST: ld1b    { z0.b }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x40,0x00,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 00 a4 <unknown>

ld1b    { z5.h }, p3/z, [x17, x16]
// CHECK-INST: ld1b    { z5.h }, p3/z, [x17, x16]
// CHECK-ENCODING: [0x25,0x4e,0x30,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 25 4e 30 a4 <unknown>

ld1b    { z21.s }, p5/z, [x10, x21]
// CHECK-INST: ld1b    { z21.s }, p5/z, [x10, x21]
// CHECK-ENCODING: [0x55,0x55,0x55,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 55 a4 <unknown>

ld1b    { z23.d }, p3/z, [x13, x8]
// CHECK-INST: ld1b    { z23.d }, p3/z, [x13, x8]
// CHECK-ENCODING: [0xb7,0x4d,0x68,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 4d 68 a4 <unknown>
