// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld2b    { z0.b, z1.b }, p0/z, [x0, x0]
// CHECK-INST: ld2b    { z0.b, z1.b }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0xc0,0x20,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 20 a4 <unknown>

ld2b    { z5.b, z6.b }, p3/z, [x17, x16]
// CHECK-INST: ld2b    { z5.b, z6.b }, p3/z, [x17, x16]
// CHECK-ENCODING: [0x25,0xce,0x30,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 25 ce 30 a4 <unknown>

ld2b    { z0.b, z1.b }, p0/z, [x0]
// CHECK-INST: ld2b    { z0.b, z1.b }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x20,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 20 a4 <unknown>

ld2b    { z23.b, z24.b }, p3/z, [x13, #-16, mul vl]
// CHECK-INST: ld2b    { z23.b, z24.b }, p3/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x28,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 ed 28 a4 <unknown>

ld2b    { z21.b, z22.b }, p5/z, [x10, #10, mul vl]
// CHECK-INST: ld2b    { z21.b, z22.b }, p5/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x25,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 f5 25 a4 <unknown>
