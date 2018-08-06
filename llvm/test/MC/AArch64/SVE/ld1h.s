// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld1h     z0.h, p0/z, [x0]
// CHECK-INST: ld1h     { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 a0 a4 <unknown>

ld1h     z0.s, p0/z, [x0]
// CHECK-INST: ld1h     { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xc0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 c0 a4 <unknown>

ld1h     z0.d, p0/z, [x0]
// CHECK-INST: ld1h     { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xe0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 e0 a4 <unknown>

ld1h    { z0.h }, p0/z, [x0]
// CHECK-INST: ld1h    { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 a0 a4 <unknown>

ld1h    { z0.s }, p0/z, [x0]
// CHECK-INST: ld1h    { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xc0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 c0 a4 <unknown>

ld1h    { z0.d }, p0/z, [x0]
// CHECK-INST: ld1h    { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xe0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 e0 a4 <unknown>

ld1h    { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1h    { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xaf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf af a4 <unknown>

ld1h    { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1h    { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xa5,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 a5 a4 <unknown>

ld1h    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1h    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xcf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf cf a4 <unknown>

ld1h    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1h    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xc5,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 c5 a4 <unknown>

ld1h    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1h    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xef,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf ef a4 <unknown>

ld1h    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1h    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xe5,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 e5 a4 <unknown>

ld1h    { z5.h }, p3/z, [sp, x16, lsl #1]
// CHECK-INST: ld1h    { z5.h }, p3/z, [sp, x16, lsl #1]
// CHECK-ENCODING: [0xe5,0x4f,0xb0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 4f b0 a4 <unknown>

ld1h    { z5.h }, p3/z, [x17, x16, lsl #1]
// CHECK-INST: ld1h    { z5.h }, p3/z, [x17, x16, lsl #1]
// CHECK-ENCODING: [0x25,0x4e,0xb0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 25 4e b0 a4 <unknown>

ld1h    { z21.s }, p5/z, [x10, x21, lsl #1]
// CHECK-INST: ld1h    { z21.s }, p5/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x55,0x55,0xd5,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 d5 a4 <unknown>

ld1h    { z23.d }, p3/z, [x13, x8, lsl #1]
// CHECK-INST: ld1h    { z23.d }, p3/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xb7,0x4d,0xe8,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 4d e8 a4 <unknown>

ld1h    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ld1h    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x40,0x80,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 80 84 <unknown>

ld1h    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ld1h    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x40,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 c0 84 <unknown>

ld1h    { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-INST: ld1h    { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-ENCODING: [0xff,0x5f,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 5f bf 84 <unknown>

ld1h    { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-INST: ld1h    { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-ENCODING: [0xff,0x5f,0xff,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 5f ff 84 <unknown>

ld1h    { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ld1h    { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xdf,0xdf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df df c4 <unknown>

ld1h    { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-INST: ld1h    { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-ENCODING: [0xb7,0xcd,0xe8,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 cd e8 c4 <unknown>

ld1h    { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ld1h    { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x55,0x95,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 95 c4 <unknown>

ld1h    { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ld1h    { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x55,0xd5,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 d5 c4 <unknown>

ld1h    { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-INST: ld1h    { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-ENCODING: [0x00,0x40,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 a0 c4 <unknown>

ld1h    { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-INST: ld1h    { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-ENCODING: [0x00,0x40,0xe0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 e0 c4 <unknown>

ld1h    { z31.s }, p7/z, [z31.s, #62]
// CHECK-INST: ld1h    { z31.s }, p7/z, [z31.s, #62]
// CHECK-ENCODING: [0xff,0xdf,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df bf 84 <unknown>

ld1h    { z0.s }, p0/z, [z0.s]
// CHECK-INST: ld1h    { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0xc0,0xa0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 a0 84 <unknown>

ld1h    { z31.d }, p7/z, [z31.d, #62]
// CHECK-INST: ld1h    { z31.d }, p7/z, [z31.d, #62]
// CHECK-ENCODING: [0xff,0xdf,0xbf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df bf c4 <unknown>

ld1h    { z0.d }, p0/z, [z0.d]
// CHECK-INST: ld1h    { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xc0,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 a0 c4 <unknown>
