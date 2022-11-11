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

st2d    { z0.d, z1.d }, p0, [x0, x0, lsl #3]
// CHECK-INST: st2d    { z0.d, z1.d }, p0, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0x60,0xa0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5a06000 <unknown>

st2d    { z5.d, z6.d }, p3, [x17, x16, lsl #3]
// CHECK-INST: st2d    { z5.d, z6.d }, p3, [x17, x16, lsl #3]
// CHECK-ENCODING: [0x25,0x6e,0xb0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5b06e25 <unknown>

st2d    { z0.d, z1.d }, p0, [x0]
// CHECK-INST: st2d    { z0.d, z1.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xb0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5b0e000 <unknown>

st2d    { z23.d, z24.d }, p3, [x13, #-16, mul vl]
// CHECK-INST: st2d    { z23.d, z24.d }, p3, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0xb8,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5b8edb7 <unknown>

st2d    { z21.d, z22.d }, p5, [x10, #10, mul vl]
// CHECK-INST: st2d    { z21.d, z22.d }, p5, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xb5,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5b5f555 <unknown>
