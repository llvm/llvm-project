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

st1h    z0.h, p0, [x0]
// CHECK-INST: st1h    { z0.h }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xa0,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4a0e000 <unknown>

st1h    z0.s, p0, [x0]
// CHECK-INST: st1h    { z0.s }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4c0e000 <unknown>

st1h    z0.d, p0, [x0]
// CHECK-INST: st1h    { z0.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4e0e000 <unknown>

st1h    { z0.h }, p0, [x0]
// CHECK-INST: st1h    { z0.h }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xa0,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4a0e000 <unknown>

st1h    { z0.s }, p0, [x0]
// CHECK-INST: st1h    { z0.s }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4c0e000 <unknown>

st1h    { z0.d }, p0, [x0]
// CHECK-INST: st1h    { z0.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4e0e000 <unknown>

st1h    { z31.h }, p7, [sp, #-1, mul vl]
// CHECK-INST: st1h    { z31.h }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0xaf,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4afffff <unknown>

st1h    { z21.h }, p5, [x10, #5, mul vl]
// CHECK-INST: st1h    { z21.h }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xa5,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4a5f555 <unknown>

st1h    { z31.s }, p7, [sp, #-1, mul vl]
// CHECK-INST: st1h    { z31.s }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0xcf,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4cfffff <unknown>

st1h    { z21.s }, p5, [x10, #5, mul vl]
// CHECK-INST: st1h    { z21.s }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xc5,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4c5f555 <unknown>

st1h    { z21.d }, p5, [x10, #5, mul vl]
// CHECK-INST: st1h    { z21.d }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xe5,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4e5f555 <unknown>

st1h    { z31.d }, p7, [sp, #-1, mul vl]
// CHECK-INST: st1h    { z31.d }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0xef,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4efffff <unknown>

st1h    { z0.h }, p0, [x0, x0, lsl #1]
// CHECK-INST: st1h    { z0.h }, p0, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x40,0xa0,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4a04000 <unknown>

st1h    { z0.s }, p0, [x0, x0, lsl #1]
// CHECK-INST: st1h    { z0.s }, p0, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x40,0xc0,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4c04000 <unknown>

st1h    { z0.d }, p0, [x0, x0, lsl #1]
// CHECK-INST: st1h    { z0.d }, p0, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x40,0xe0,0xe4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4e04000 <unknown>
