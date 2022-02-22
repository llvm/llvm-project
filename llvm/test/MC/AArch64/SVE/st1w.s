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

st1w    z0.s, p0, [x0]
// CHECK-INST: st1w    { z0.s }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x40,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 40 e5 <unknown>

st1w    z0.d, p0, [x0]
// CHECK-INST: st1w    { z0.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x60,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 60 e5 <unknown>

st1w    { z0.s }, p0, [x0]
// CHECK-INST: st1w    { z0.s }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x40,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 40 e5 <unknown>

st1w    { z0.d }, p0, [x0]
// CHECK-INST: st1w    { z0.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x60,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 60 e5 <unknown>

st1w    { z31.s }, p7, [sp, #-1, mul vl]
// CHECK-INST: st1w    { z31.s }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0x4f,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 4f e5 <unknown>

st1w    { z21.s }, p5, [x10, #5, mul vl]
// CHECK-INST: st1w    { z21.s }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x45,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 f5 45 e5 <unknown>

st1w    { z31.d }, p7, [sp, #-1, mul vl]
// CHECK-INST: st1w    { z31.d }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0x6f,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 6f e5 <unknown>

st1w    { z21.d }, p5, [x10, #5, mul vl]
// CHECK-INST: st1w    { z21.d }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x65,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 f5 65 e5 <unknown>

st1w    { z0.s }, p0, [x0, x0, lsl #2]
// CHECK-INST: st1w    { z0.s }, p0, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x40,0x40,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 40 40 e5 <unknown>

st1w    { z0.d }, p0, [x0, x0, lsl #2]
// CHECK-INST: st1w    { z0.d }, p0, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x40,0x60,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 40 60 e5 <unknown>
