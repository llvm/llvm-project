// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld1h     z0.h, p0/z, [x0]
// CHECK-INST: ld1h     { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 a0 a4 <unknown>

ld1h     z0.s, p0/z, [x0]
// CHECK-INST: ld1h     { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xc0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 c0 a4 <unknown>

ld1h     z0.d, p0/z, [x0]
// CHECK-INST: ld1h     { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xe0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 e0 a4 <unknown>

ld1h    { z0.h }, p0/z, [x0]
// CHECK-INST: ld1h    { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 a0 a4 <unknown>

ld1h    { z0.s }, p0/z, [x0]
// CHECK-INST: ld1h    { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xc0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 c0 a4 <unknown>

ld1h    { z0.d }, p0/z, [x0]
// CHECK-INST: ld1h    { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xe0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 e0 a4 <unknown>

ld1h    { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1h    { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xaf,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf af a4 <unknown>

ld1h    { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1h    { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xa5,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 b5 a5 a4 <unknown>

ld1h    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1h    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xcf,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf cf a4 <unknown>

ld1h    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1h    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xc5,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 b5 c5 a4 <unknown>

ld1h    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1h    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xef,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf ef a4 <unknown>

ld1h    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1h    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xe5,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 b5 e5 a4 <unknown>

ld1h    { z5.h }, p3/z, [sp, x16, lsl #1]
// CHECK-INST: ld1h    { z5.h }, p3/z, [sp, x16, lsl #1]
// CHECK-ENCODING: [0xe5,0x4f,0xb0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5 4f b0 a4 <unknown>

ld1h    { z5.h }, p3/z, [x17, x16, lsl #1]
// CHECK-INST: ld1h    { z5.h }, p3/z, [x17, x16, lsl #1]
// CHECK-ENCODING: [0x25,0x4e,0xb0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25 4e b0 a4 <unknown>

ld1h    { z21.s }, p5/z, [x10, x21, lsl #1]
// CHECK-INST: ld1h    { z21.s }, p5/z, [x10, x21, lsl #1]
// CHECK-ENCODING: [0x55,0x55,0xd5,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 55 d5 a4 <unknown>

ld1h    { z23.d }, p3/z, [x13, x8, lsl #1]
// CHECK-INST: ld1h    { z23.d }, p3/z, [x13, x8, lsl #1]
// CHECK-ENCODING: [0xb7,0x4d,0xe8,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 4d e8 a4 <unknown>
