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

st4b    { z0.b, z1.b, z2.b, z3.b }, p0, [x0, x0]
// CHECK-INST: st4b    { z0.b - z3.b }, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x60,0x60,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4606000 <unknown>

st4b    { z5.b, z6.b, z7.b, z8.b }, p3, [x17, x16]
// CHECK-INST: st4b    { z5.b - z8.b }, p3, [x17, x16]
// CHECK-ENCODING: [0x25,0x6e,0x70,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4706e25 <unknown>

st4b    { z0.b, z1.b, z2.b, z3.b }, p0, [x0]
// CHECK-INST: st4b    { z0.b - z3.b }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x70,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e470e000 <unknown>

st4b    { z23.b, z24.b, z25.b, z26.b }, p3, [x13, #-32, mul vl]
// CHECK-INST: st4b    { z23.b - z26.b }, p3, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x78,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e478edb7 <unknown>

st4b    { z21.b, z22.b, z23.b, z24.b }, p5, [x10, #20, mul vl]
// CHECK-INST: st4b    { z21.b - z24.b }, p5, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x75,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e475f555 <unknown>
