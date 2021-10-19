// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

ld1b    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ld1b    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x40,0x00,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 00 84 <unknown>

ld1b    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ld1b    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x40,0x40,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 40 84 <unknown>

ld1b    { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ld1b    { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xdf,0x5f,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df 5f c4 <unknown>

ld1b    { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ld1b    { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x55,0x15,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 15 c4 <unknown>

ld1b    { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ld1b    { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x55,0x55,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 55 55 c4 <unknown>

ld1b    { z31.s }, p7/z, [z31.s, #31]
// CHECK-INST: ld1b    { z31.s }, p7/z, [z31.s, #31]
// CHECK-ENCODING: [0xff,0xdf,0x3f,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df 3f 84 <unknown>

ld1b    { z0.s }, p0/z, [z0.s]
// CHECK-INST: ld1b    { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0xc0,0x20,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 20 84 <unknown>

ld1b    { z31.d }, p7/z, [z31.d, #31]
// CHECK-INST: ld1b    { z31.d }, p7/z, [z31.d, #31]
// CHECK-ENCODING: [0xff,0xdf,0x3f,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df 3f c4 <unknown>

ld1b    { z0.d }, p0/z, [z0.d]
// CHECK-INST: ld1b    { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xc0,0x20,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 20 c4 <unknown>
