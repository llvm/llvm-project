// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld3w    { z0.s, z1.s, z2.s }, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ld3w    { z0.s, z1.s, z2.s }, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0xc0,0x40,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 40 a5 <unknown>

ld3w    { z5.s, z6.s, z7.s }, p3/z, [x17, x16, lsl #2]
// CHECK-INST: ld3w    { z5.s, z6.s, z7.s }, p3/z, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0xce,0x50,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 25 ce 50 a5 <unknown>

ld3w    { z0.s, z1.s, z2.s }, p0/z, [x0]
// CHECK-INST: ld3w    { z0.s, z1.s, z2.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x40,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 40 a5 <unknown>

ld3w    { z23.s, z24.s, z25.s }, p3/z, [x13, #-24, mul vl]
// CHECK-INST: ld3w    { z23.s, z24.s, z25.s }, p3/z, [x13, #-24, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x48,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 ed 48 a5 <unknown>

ld3w    { z21.s, z22.s, z23.s }, p5/z, [x10, #15, mul vl]
// CHECK-INST: ld3w    { z21.s, z22.s, z23.s }, p5/z, [x10, #15, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x45,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 f5 45 a5 <unknown>
