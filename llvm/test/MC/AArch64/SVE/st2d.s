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

st2d    { z0.d, z1.d }, p0, [x0, x0, lsl #3]
// CHECK-INST: st2d    { z0.d, z1.d }, p0, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0x60,0xa0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 60 a0 e5 <unknown>

st2d    { z5.d, z6.d }, p3, [x17, x16, lsl #3]
// CHECK-INST: st2d    { z5.d, z6.d }, p3, [x17, x16, lsl #3]
// CHECK-ENCODING: [0x25,0x6e,0xb0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25 6e b0 e5 <unknown>

st2d    { z0.d, z1.d }, p0, [x0]
// CHECK-INST: st2d    { z0.d, z1.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xb0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 b0 e5 <unknown>

st2d    { z23.d, z24.d }, p3, [x13, #-16, mul vl]
// CHECK-INST: st2d    { z23.d, z24.d }, p3, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0xb8,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 ed b8 e5 <unknown>

st2d    { z21.d, z22.d }, p5, [x10, #10, mul vl]
// CHECK-INST: st2d    { z21.d, z22.d }, p5, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xb5,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 f5 b5 e5 <unknown>
