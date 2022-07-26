// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

st3d    { z0.d, z1.d, z2.d }, p0, [x0, x0, lsl #3]
// CHECK-INST: st3d    { z0.d, z1.d, z2.d }, p0, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0x60,0xc0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5c06000 <unknown>

st3d    { z5.d, z6.d, z7.d }, p3, [x17, x16, lsl #3]
// CHECK-INST: st3d    { z5.d, z6.d, z7.d }, p3, [x17, x16, lsl #3]
// CHECK-ENCODING: [0x25,0x6e,0xd0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5d06e25 <unknown>

st3d    { z0.d, z1.d, z2.d }, p0, [x0]
// CHECK-INST: st3d    { z0.d, z1.d, z2.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xd0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5d0e000 <unknown>

st3d    { z23.d, z24.d, z25.d }, p3, [x13, #-24, mul vl]
// CHECK-INST: st3d    { z23.d, z24.d, z25.d }, p3, [x13, #-24, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0xd8,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5d8edb7 <unknown>

st3d    { z21.d, z22.d, z23.d }, p5, [x10, #15, mul vl]
// CHECK-INST: st3d    { z21.d, z22.d, z23.d }, p5, [x10, #15, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xd5,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5d5f555 <unknown>
