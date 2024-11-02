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

st1d    z0.d, p0, [x0]
// CHECK-INST: st1d    { z0.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xe0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5e0e000 <unknown>

st1d    { z0.d }, p0, [x0]
// CHECK-INST: st1d    { z0.d }, p0, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xe0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5e0e000 <unknown>

st1d    { z31.d }, p7, [sp, #-1, mul vl]
// CHECK-INST: st1d    { z31.d }, p7, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xff,0xef,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5efffff <unknown>

st1d    { z21.d }, p5, [x10, #5, mul vl]
// CHECK-INST: st1d    { z21.d }, p5, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xe5,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5e5f555 <unknown>

st1d    { z0.d }, p0, [x0, x0, lsl #3]
// CHECK-INST: st1d    { z0.d }, p0, [x0, x0, lsl #3]
// CHECK-ENCODING: [0x00,0x40,0xe0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5e04000 <unknown>
