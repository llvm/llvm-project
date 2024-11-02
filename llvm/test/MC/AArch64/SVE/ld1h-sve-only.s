// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Test instruction variants that aren't legal in streaming mode.

ld1h    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ld1h    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x40,0x80,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84804000 <unknown>

ld1h    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ld1h    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x40,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84c04000 <unknown>

ld1h    { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-INST: ld1h    { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-ENCODING: [0xff,0x5f,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84bf5fff <unknown>

ld1h    { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-INST: ld1h    { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-ENCODING: [0xff,0x5f,0xff,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84ff5fff <unknown>

ld1h    { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ld1h    { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xdf,0xdf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4dfdfff <unknown>

ld1h    { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-INST: ld1h    { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-ENCODING: [0xb7,0xcd,0xe8,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4e8cdb7 <unknown>

ld1h    { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ld1h    { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x55,0x95,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4955555 <unknown>

ld1h    { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ld1h    { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x55,0xd5,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4d55555 <unknown>

ld1h    { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-INST: ld1h    { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-ENCODING: [0x00,0x40,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4a04000 <unknown>

ld1h    { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-INST: ld1h    { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-ENCODING: [0x00,0x40,0xe0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4e04000 <unknown>

ld1h    { z31.s }, p7/z, [z31.s, #62]
// CHECK-INST: ld1h    { z31.s }, p7/z, [z31.s, #62]
// CHECK-ENCODING: [0xff,0xdf,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84bfdfff <unknown>

ld1h    { z0.s }, p0/z, [z0.s]
// CHECK-INST: ld1h    { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0xc0,0xa0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84a0c000 <unknown>

ld1h    { z31.d }, p7/z, [z31.d, #62]
// CHECK-INST: ld1h    { z31.d }, p7/z, [z31.d, #62]
// CHECK-ENCODING: [0xff,0xdf,0xbf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4bfdfff <unknown>

ld1h    { z0.d }, p0/z, [z0.d]
// CHECK-INST: ld1h    { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xc0,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4a0c000 <unknown>
