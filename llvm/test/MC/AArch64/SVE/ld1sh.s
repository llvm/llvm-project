// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld1sh   z0.s, p0/z, [x0]
// CHECK-INST: ld1sh   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x20,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 20 a5 <unknown>

ld1sh   z0.d, p0/z, [x0]
// CHECK-INST: ld1sh   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x00,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 00 a5 <unknown>

ld1sh   { z0.s }, p0/z, [x0]
// CHECK-INST: ld1sh   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x20,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 20 a5 <unknown>

ld1sh   { z0.d }, p0/z, [x0]
// CHECK-INST: ld1sh   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x00,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 00 a5 <unknown>

ld1sh   { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1sh   { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x2f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 2f a5 <unknown>

ld1sh   { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1sh   { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x25,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 25 a5 <unknown>

ld1sh   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1sh   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x0f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 0f a5 <unknown>

ld1sh   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1sh   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x05,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 05 a5 <unknown>

ld1sh    { z21.s }, p5/z, [sp, x21, lsl #1]
// CHECK-INST: ld1sh    { z21.s }, p5/z, [sp, x21, lsl #1]
// CHECK-ENCODING: [0xf5,0x57,0x35,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f5 57 35 a5 <unknown>

ld1sh    { z21.s }, p5/z, [x10, x21, lsl #1]
// CHECK-INST: ld1sh    { z21.s }, p5/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x55,0x55,0x35,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 35 a5 <unknown>

ld1sh    { z23.d }, p3/z, [x13, x8, lsl #1]
// CHECK-INST: ld1sh    { z23.d }, p3/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xb7,0x4d,0x08,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 4d 08 a5 <unknown>

ld1sh   { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ld1sh   { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x00,0x80,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 80 84 <unknown>

ld1sh   { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ld1sh   { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x00,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 c0 84 <unknown>

ld1sh   { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-INST: ld1sh   { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-ENCODING: [0xff,0x1f,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f bf 84 <unknown>

ld1sh   { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-INST: ld1sh   { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-ENCODING: [0xff,0x1f,0xff,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f ff 84 <unknown>

ld1sh   { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ld1sh   { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0x9f,0xdf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f df c4 <unknown>

ld1sh   { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-INST: ld1sh   { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-ENCODING: [0xb7,0x8d,0xe8,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 8d e8 c4 <unknown>

ld1sh   { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ld1sh   { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x15,0x95,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 15 95 c4 <unknown>

ld1sh   { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ld1sh   { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x15,0xd5,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 15 d5 c4 <unknown>

ld1sh   { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-INST: ld1sh   { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-ENCODING: [0x00,0x00,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 a0 c4 <unknown>

ld1sh   { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-INST: ld1sh   { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-ENCODING: [0x00,0x00,0xe0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 e0 c4 <unknown>

ld1sh   { z31.s }, p7/z, [z31.s, #62]
// CHECK-INST: ld1sh   { z31.s }, p7/z, [z31.s, #62]
// CHECK-ENCODING: [0xff,0x9f,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f bf 84 <unknown>

ld1sh   { z0.s }, p0/z, [z0.s]
// CHECK-INST: ld1sh   { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0x80,0xa0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 a0 84 <unknown>

ld1sh   { z31.d }, p7/z, [z31.d, #62]
// CHECK-INST: ld1sh   { z31.d }, p7/z, [z31.d, #62]
// CHECK-ENCODING: [0xff,0x9f,0xbf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f bf c4 <unknown>

ld1sh   { z0.d }, p0/z, [z0.d]
// CHECK-INST: ld1sh   { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0x80,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 a0 c4 <unknown>
