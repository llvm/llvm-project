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

ld1d    { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ld1d    { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xdf,0xdf,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df df c5 <unknown>

ld1d    { z23.d }, p3/z, [x13, z8.d, lsl #3]
// CHECK-INST: ld1d    { z23.d }, p3/z, [x13, z8.d, lsl #3]
// CHECK-ENCODING: [0xb7,0xcd,0xe8,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 cd e8 c5 <unknown>

ld1d    { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ld1d    { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x55,0x95,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 95 c5 <unknown>

ld1d    { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ld1d    { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x55,0xd5,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 d5 c5 <unknown>

ld1d    { z0.d }, p0/z, [x0, z0.d, uxtw #3]
// CHECK-INST: ld1d    { z0.d }, p0/z, [x0, z0.d, uxtw #3]
// CHECK-ENCODING: [0x00,0x40,0xa0,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 a0 c5 <unknown>

ld1d    { z0.d }, p0/z, [x0, z0.d, sxtw #3]
// CHECK-INST: ld1d    { z0.d }, p0/z, [x0, z0.d, sxtw #3]
// CHECK-ENCODING: [0x00,0x40,0xe0,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 e0 c5 <unknown>

ld1d    { z31.d }, p7/z, [z31.d, #248]
// CHECK-INST: ld1d    { z31.d }, p7/z, [z31.d, #248]
// CHECK-ENCODING: [0xff,0xdf,0xbf,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df bf c5 <unknown>

ld1d    { z0.d }, p0/z, [z0.d]
// CHECK-INST: ld1d    { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xc0,0xa0,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 a0 c5 <unknown>
