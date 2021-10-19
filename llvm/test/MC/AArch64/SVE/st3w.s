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

st3w    { z0.s, z1.s, z2.s }, p0, [x0, x0, lsl #2]
// CHECK-INST: st3w    { z0.s, z1.s, z2.s }, p0, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x60,0x40,0xe5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 60 40 e5 <unknown>

st3w    { z5.s, z6.s, z7.s }, p3, [x17, x16, lsl #2]
// CHECK-INST: st3w    { z5.s, z6.s, z7.s }, p3, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0x6e,0x50,0xe5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 25 6e 50 e5 <unknown>

st3w    { z0.s, z1.s, z2.s }, p0, [x0]
// CHECK-INST: st3w    { z0.s, z1.s, z2.s }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x50,0xe5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 e0 50 e5 <unknown>

st3w    { z23.s, z24.s, z25.s }, p3, [x13, #-24, mul vl]
// CHECK-INST: st3w    { z23.s, z24.s, z25.s }, p3, [x13, #-24, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x58,0xe5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: b7 ed 58 e5 <unknown>

st3w    { z21.s, z22.s, z23.s }, p5, [x10, #15, mul vl]
// CHECK-INST: st3w    { z21.s, z22.s, z23.s }, p5, [x10, #15, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x55,0xe5]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 55 f5 55 e5 <unknown>
