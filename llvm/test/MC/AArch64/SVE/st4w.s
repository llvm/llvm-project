// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

st4w    { z0.s, z1.s, z2.s, z3.s }, p0, [x0, x0, lsl #2]
// CHECK-INST: st4w    { z0.s - z3.s }, p0, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x60,0x60,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5606000 <unknown>

st4w    { z5.s, z6.s, z7.s, z8.s }, p3, [x17, x16, lsl #2]
// CHECK-INST: st4w    { z5.s - z8.s }, p3, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0x6e,0x70,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5706e25 <unknown>

st4w    { z0.s, z1.s, z2.s, z3.s }, p0, [x0]
// CHECK-INST: st4w    { z0.s - z3.s }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x70,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e570e000 <unknown>

st4w    { z23.s, z24.s, z25.s, z26.s }, p3, [x13, #-32, mul vl]
// CHECK-INST: st4w    { z23.s - z26.s }, p3, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x78,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e578edb7 <unknown>

st4w    { z21.s, z22.s, z23.s, z24.s }, p5, [x10, #20, mul vl]
// CHECK-INST: st4w    { z21.s - z24.s }, p5, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x75,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e575f555 <unknown>
