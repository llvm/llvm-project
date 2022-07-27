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

ldff1sh { z31.s }, p7/z, [sp]
// CHECK-INST: ldff1sh { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x3f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a53f7fff <unknown>

ldff1sh { z31.d }, p7/z, [sp]
// CHECK-INST: ldff1sh { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x1f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a51f7fff <unknown>

ldff1sh { z31.s }, p7/z, [sp, xzr, lsl #1]
// CHECK-INST: ldff1sh { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x3f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a53f7fff <unknown>

ldff1sh { z31.d }, p7/z, [sp, xzr, lsl #1]
// CHECK-INST: ldff1sh { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x1f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a51f7fff <unknown>

ldff1sh { z0.s }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldff1sh { z0.s }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0x20,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5206000 <unknown>

ldff1sh { z0.d }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldff1sh { z0.d }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0x00,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5006000 <unknown>

ldff1sh   { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ldff1sh   { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x20,0x80,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84802000 <unknown>

ldff1sh   { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ldff1sh   { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x20,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84c02000 <unknown>

ldff1sh { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-INST: ldff1sh { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-ENCODING: [0xff,0x3f,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84bf3fff <unknown>

ldff1sh { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-INST: ldff1sh { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-ENCODING: [0xff,0x3f,0xff,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84ff3fff <unknown>

ldff1sh { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ldff1sh { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xbf,0xdf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4dfbfff <unknown>

ldff1sh { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-INST: ldff1sh { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-ENCODING: [0xb7,0xad,0xe8,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4e8adb7 <unknown>

ldff1sh { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ldff1sh { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x35,0x95,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4953555 <unknown>

ldff1sh { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ldff1sh { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x35,0xd5,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4d53555 <unknown>

ldff1sh { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-INST: ldff1sh { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-ENCODING: [0x00,0x20,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4a02000 <unknown>

ldff1sh { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-INST: ldff1sh { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-ENCODING: [0x00,0x20,0xe0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4e02000 <unknown>

ldff1sh { z31.s }, p7/z, [z31.s, #62]
// CHECK-INST: ldff1sh { z31.s }, p7/z, [z31.s, #62]
// CHECK-ENCODING: [0xff,0xbf,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84bfbfff <unknown>

ldff1sh { z0.s }, p0/z, [z0.s]
// CHECK-INST: ldff1sh { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84a0a000 <unknown>

ldff1sh { z31.d }, p7/z, [z31.d, #62]
// CHECK-INST: ldff1sh { z31.d }, p7/z, [z31.d, #62]
// CHECK-ENCODING: [0xff,0xbf,0xbf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4bfbfff <unknown>

ldff1sh { z0.d }, p0/z, [z0.d]
// CHECK-INST: ldff1sh { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4a0a000 <unknown>
