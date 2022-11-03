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

ldff1h  { z31.h }, p7/z, [sp]
// CHECK-INST: ldff1h  { z31.h }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xbf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4bf7fff <unknown>

ldff1h  { z31.s }, p7/z, [sp]
// CHECK-INST: ldff1h  { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xdf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4df7fff <unknown>

ldff1h  { z31.d }, p7/z, [sp]
// CHECK-INST: ldff1h  { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xff,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4ff7fff <unknown>

ldff1h  { z31.h }, p7/z, [sp, xzr, lsl #1]
// CHECK-INST: ldff1h  { z31.h }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xbf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4bf7fff <unknown>

ldff1h  { z31.s }, p7/z, [sp, xzr, lsl #1]
// CHECK-INST: ldff1h  { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xdf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4df7fff <unknown>

ldff1h  { z31.d }, p7/z, [sp, xzr, lsl #1]
// CHECK-INST: ldff1h  { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xff,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4ff7fff <unknown>

ldff1h  { z0.h }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldff1h  { z0.h }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0xa0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4a06000 <unknown>

ldff1h  { z0.s }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldff1h  { z0.s }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0xc0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4c06000 <unknown>

ldff1h  { z0.d }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldff1h  { z0.d }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0xe0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4e06000 <unknown>

ldff1h    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ldff1h    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x60,0x80,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84806000 <unknown>

ldff1h    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ldff1h    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x60,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84c06000 <unknown>

ldff1h  { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-INST: ldff1h  { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-ENCODING: [0xff,0x7f,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84bf7fff <unknown>

ldff1h  { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-INST: ldff1h  { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-ENCODING: [0xff,0x7f,0xff,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84ff7fff <unknown>

ldff1h  { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ldff1h  { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xff,0xdf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4dfffff <unknown>

ldff1h  { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-INST: ldff1h  { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-ENCODING: [0xb7,0xed,0xe8,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4e8edb7 <unknown>

ldff1h  { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ldff1h  { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x75,0x95,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4957555 <unknown>

ldff1h  { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ldff1h  { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x75,0xd5,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4d57555 <unknown>

ldff1h  { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-INST: ldff1h  { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-ENCODING: [0x00,0x60,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4a06000 <unknown>

ldff1h  { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-INST: ldff1h  { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-ENCODING: [0x00,0x60,0xe0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4e06000 <unknown>

ldff1h  { z31.s }, p7/z, [z31.s, #62]
// CHECK-INST: ldff1h  { z31.s }, p7/z, [z31.s, #62]
// CHECK-ENCODING: [0xff,0xff,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84bfffff <unknown>

ldff1h  { z0.s }, p0/z, [z0.s]
// CHECK-INST: ldff1h  { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0xe0,0xa0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84a0e000 <unknown>

ldff1h  { z31.d }, p7/z, [z31.d, #62]
// CHECK-INST: ldff1h  { z31.d }, p7/z, [z31.d, #62]
// CHECK-ENCODING: [0xff,0xff,0xbf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4bfffff <unknown>

ldff1h  { z0.d }, p0/z, [z0.d]
// CHECK-INST: ldff1h  { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xe0,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4a0e000 <unknown>
