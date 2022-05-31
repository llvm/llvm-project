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

ldnt1h  z0.h, p0/z, [x0]
// CHECK-INST: ldnt1h  { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x80,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 80 a4 <unknown>

ldnt1h  { z0.h }, p0/z, [x0]
// CHECK-INST: ldnt1h  { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x80,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 80 a4 <unknown>

ldnt1h  { z23.h }, p3/z, [x13, #-8, mul vl]
// CHECK-INST: ldnt1h  { z23.h }, p3/z, [x13, #-8, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x88,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 ed 88 a4 <unknown>

ldnt1h  { z21.h }, p5/z, [x10, #7, mul vl]
// CHECK-INST: ldnt1h  { z21.h }, p5/z, [x10, #7, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x87,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 f5 87 a4 <unknown>

ldnt1h  { z0.h }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldnt1h  { z0.h }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0xc0,0x80,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 80 a4 <unknown>
