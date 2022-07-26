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

st2b    { z0.b, z1.b }, p0, [x0, x0]
// CHECK-INST: st2b    { z0.b, z1.b }, p0, [x0, x0]
// CHECK-ENCODING: [0x00,0x60,0x20,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4206000 <unknown>

st2b    { z5.b, z6.b }, p3, [x17, x16]
// CHECK-INST: st2b    { z5.b, z6.b }, p3, [x17, x16]
// CHECK-ENCODING: [0x25,0x6e,0x30,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4306e25 <unknown>

st2b    { z0.b, z1.b }, p0, [x0]
// CHECK-INST: st2b    { z0.b, z1.b }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x30,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e430e000 <unknown>

st2b    { z23.b, z24.b }, p3, [x13, #-16, mul vl]
// CHECK-INST: st2b    { z23.b, z24.b }, p3, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x38,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e438edb7 <unknown>

st2b    { z21.b, z22.b }, p5, [x10, #10, mul vl]
// CHECK-INST: st2b    { z21.b, z22.b }, p5, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x35,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e435f555 <unknown>
