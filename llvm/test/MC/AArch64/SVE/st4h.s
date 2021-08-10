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

st4h    { z0.h, z1.h, z2.h, z3.h }, p0, [x0, x0, lsl #1]
// CHECK-INST: st4h    { z0.h, z1.h, z2.h, z3.h }, p0, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0xe0,0xe4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 60 e0 e4 <unknown>

st4h    { z5.h, z6.h, z7.h, z8.h }, p3, [x17, x16, lsl #1]
// CHECK-INST: st4h    { z5.h, z6.h, z7.h, z8.h }, p3, [x17, x16, lsl #1]
// CHECK-ENCODING: [0x25,0x6e,0xf0,0xe4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 25 6e f0 e4 <unknown>

st4h    { z0.h, z1.h, z2.h, z3.h }, p0, [x0]
// CHECK-INST: st4h    { z0.h, z1.h, z2.h, z3.h }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xf0,0xe4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 e0 f0 e4 <unknown>

st4h    { z23.h, z24.h, z25.h, z26.h }, p3, [x13, #-32, mul vl]
// CHECK-INST: st4h    { z23.h, z24.h, z25.h, z26.h }, p3, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0xf8,0xe4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: b7 ed f8 e4 <unknown>

st4h    { z21.h, z22.h, z23.h, z24.h }, p5, [x10, #20, mul vl]
// CHECK-INST: st4h    { z21.h, z22.h, z23.h, z24.h }, p5, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xf5,0xe4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 55 f5 f5 e4 <unknown>
