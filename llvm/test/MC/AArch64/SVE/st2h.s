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

st2h    { z0.h, z1.h }, p0, [x0, x0, lsl #1]
// CHECK-INST: st2h    { z0.h, z1.h }, p0, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0xa0,0xe4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 60 a0 e4 <unknown>

st2h    { z5.h, z6.h }, p3, [x17, x16, lsl #1]
// CHECK-INST: st2h    { z5.h, z6.h }, p3, [x17, x16, lsl #1]
// CHECK-ENCODING: [0x25,0x6e,0xb0,0xe4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 25 6e b0 e4 <unknown>

st2h    { z0.h, z1.h }, p0, [x0]
// CHECK-INST: st2h    { z0.h, z1.h }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xb0,0xe4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 e0 b0 e4 <unknown>

st2h    { z23.h, z24.h }, p3, [x13, #-16, mul vl]
// CHECK-INST: st2h    { z23.h, z24.h }, p3, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0xb8,0xe4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: b7 ed b8 e4 <unknown>

st2h    { z21.h, z22.h }, p5, [x10, #10, mul vl]
// CHECK-INST: st2h    { z21.h, z22.h }, p5, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xb5,0xe4]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 55 f5 b5 e4 <unknown>
