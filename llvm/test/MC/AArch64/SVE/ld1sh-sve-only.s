// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

ld1sh   { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ld1sh   { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x00,0x80,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84800000 <unknown>

ld1sh   { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ld1sh   { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x00,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84c00000 <unknown>

ld1sh   { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-INST: ld1sh   { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-ENCODING: [0xff,0x1f,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84bf1fff <unknown>

ld1sh   { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-INST: ld1sh   { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-ENCODING: [0xff,0x1f,0xff,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84ff1fff <unknown>

ld1sh   { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ld1sh   { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0x9f,0xdf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4df9fff <unknown>

ld1sh   { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-INST: ld1sh   { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-ENCODING: [0xb7,0x8d,0xe8,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4e88db7 <unknown>

ld1sh   { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ld1sh   { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x15,0x95,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4951555 <unknown>

ld1sh   { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ld1sh   { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x15,0xd5,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4d51555 <unknown>

ld1sh   { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-INST: ld1sh   { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-ENCODING: [0x00,0x00,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4a00000 <unknown>

ld1sh   { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-INST: ld1sh   { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-ENCODING: [0x00,0x00,0xe0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4e00000 <unknown>

ld1sh   { z31.s }, p7/z, [z31.s, #62]
// CHECK-INST: ld1sh   { z31.s }, p7/z, [z31.s, #62]
// CHECK-ENCODING: [0xff,0x9f,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84bf9fff <unknown>

ld1sh   { z0.s }, p0/z, [z0.s]
// CHECK-INST: ld1sh   { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0x80,0xa0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84a08000 <unknown>

ld1sh   { z31.d }, p7/z, [z31.d, #62]
// CHECK-INST: ld1sh   { z31.d }, p7/z, [z31.d, #62]
// CHECK-ENCODING: [0xff,0x9f,0xbf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4bf9fff <unknown>

ld1sh   { z0.d }, p0/z, [z0.d]
// CHECK-INST: ld1sh   { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0x80,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4a08000 <unknown>
