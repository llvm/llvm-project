// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

st3b    { z0.b, z1.b, z2.b }, p0, [x0, x0]
// CHECK-INST: st3b    { z0.b, z1.b, z2.b }, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x60,0x40,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 40 e4 <unknown>

st3b    { z5.b, z6.b, z7.b }, p3, [x17, x16]
// CHECK-INST: st3b    { z5.b, z6.b, z7.b }, p3, [x17, x16]
// CHECK-ENCODING: [0x25,0x6e,0x50,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 25 6e 50 e4 <unknown>

st3b    { z0.b, z1.b, z2.b }, p0, [x0]
// CHECK-INST: st3b    { z0.b, z1.b, z2.b }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x50,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 50 e4 <unknown>

st3b    { z23.b, z24.b, z25.b }, p3, [x13, #-24, mul vl]
// CHECK-INST: st3b    { z23.b, z24.b, z25.b }, p3, [x13, #-24, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x58,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 ed 58 e4 <unknown>

st3b    { z21.b, z22.b, z23.b }, p5, [x10, #15, mul vl]
// CHECK-INST: st3b    { z21.b, z22.b, z23.b }, p5, [x10, #15, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x55,0xe4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 f5 55 e4 <unknown>
