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

ld1sw   z0.d, p0/z, [x0]
// CHECK-INST: ld1sw   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x80,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 80 a4 <unknown>

ld1sw   { z0.d }, p0/z, [x0]
// CHECK-INST: ld1sw   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x80,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 80 a4 <unknown>

ld1sw   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1sw   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x8f,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf 8f a4 <unknown>

ld1sw   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1sw   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x85,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 b5 85 a4 <unknown>

ld1sw    { z23.d }, p3/z, [sp, x8, lsl #2]
// CHECK-INST: ld1sw    { z23.d }, p3/z, [sp, x8, lsl #2]
// CHECK-ENCODING: [0xf7,0x4f,0x88,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: f7 4f 88 a4 <unknown>

ld1sw    { z23.d }, p3/z, [x13, x8, lsl #2]
// CHECK-INST: ld1sw    { z23.d }, p3/z, [x13, x8, lsl #2]
// CHECK-ENCODING: [0xb7,0x4d,0x88,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 4d 88 a4 <unknown>
