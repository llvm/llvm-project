// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld1d     z0.d, p0/z, [x0]
// CHECK-INST: ld1d     { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xe0,0xa5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 a0 e0 a5 <unknown>

ld1d    { z0.d }, p0/z, [x0]
// CHECK-INST: ld1d    { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xe0,0xa5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 a0 e0 a5 <unknown>

ld1d    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ld1d    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xef,0xa5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff bf ef a5 <unknown>

ld1d    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ld1d    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xe5,0xa5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 55 b5 e5 a5 <unknown>

ld1d    { z23.d }, p3/z, [sp, x8, lsl #3]
// CHECK-INST: ld1d    { z23.d }, p3/z, [sp, x8, lsl #3]
// CHECK-ENCODING: [0xf7,0x4f,0xe8,0xa5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: f7 4f e8 a5 <unknown>

ld1d    { z23.d }, p3/z, [x13, x8, lsl #3]
// CHECK-INST: ld1d    { z23.d }, p3/z, [x13, x8, lsl #3]
// CHECK-ENCODING: [0xb7,0x4d,0xe8,0xa5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: b7 4d e8 a5 <unknown>
