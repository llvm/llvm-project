// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

ld1sw   { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ld1sw   { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0x9f,0x5f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f 5f c5 <unknown>

ld1sw   { z23.d }, p3/z, [x13, z8.d, lsl #2]
// CHECK-INST: ld1sw   { z23.d }, p3/z, [x13, z8.d, lsl #2]
// CHECK-ENCODING: [0xb7,0x8d,0x68,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 8d 68 c5 <unknown>

ld1sw   { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ld1sw   { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x15,0x15,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 15 15 c5 <unknown>

ld1sw   { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ld1sw   { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x15,0x55,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 15 55 c5 <unknown>

ld1sw   { z0.d }, p0/z, [x0, z0.d, uxtw #2]
// CHECK-INST: ld1sw   { z0.d }, p0/z, [x0, z0.d, uxtw #2]
// CHECK-ENCODING: [0x00,0x00,0x20,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 20 c5 <unknown>

ld1sw   { z0.d }, p0/z, [x0, z0.d, sxtw #2]
// CHECK-INST: ld1sw   { z0.d }, p0/z, [x0, z0.d, sxtw #2]
// CHECK-ENCODING: [0x00,0x00,0x60,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 60 c5 <unknown>

ld1sw   { z31.d }, p7/z, [z31.d, #124]
// CHECK-INST: ld1sw   { z31.d }, p7/z, [z31.d, #124]
// CHECK-ENCODING: [0xff,0x9f,0x3f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 9f 3f c5 <unknown>

ld1sw   { z0.d }, p0/z, [z0.d]
// CHECK-INST: ld1sw   { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0x80,0x20,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 20 c5 <unknown>
