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

ldff1sh { z31.s }, p7/z, [sp]
// CHECK-INST: ldff1sh { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x3f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 3f a5 <unknown>

ldff1sh { z31.d }, p7/z, [sp]
// CHECK-INST: ldff1sh { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x1f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 1f a5 <unknown>

ldff1sh { z31.s }, p7/z, [sp, xzr, lsl #1]
// CHECK-INST: ldff1sh { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x3f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 3f a5 <unknown>

ldff1sh { z31.d }, p7/z, [sp, xzr, lsl #1]
// CHECK-INST: ldff1sh { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x1f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 1f a5 <unknown>

ldff1sh { z0.s }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldff1sh { z0.s }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0x20,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 20 a5 <unknown>

ldff1sh { z0.d }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldff1sh { z0.d }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0x00,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 00 a5 <unknown>

ldff1sh   { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ldff1sh   { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x20,0x80,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 80 84 <unknown>

ldff1sh   { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ldff1sh   { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x20,0xc0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 c0 84 <unknown>

ldff1sh { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-INST: ldff1sh { z31.s }, p7/z, [sp, z31.s, uxtw #1]
// CHECK-ENCODING: [0xff,0x3f,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3f bf 84 <unknown>

ldff1sh { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-INST: ldff1sh { z31.s }, p7/z, [sp, z31.s, sxtw #1]
// CHECK-ENCODING: [0xff,0x3f,0xff,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 3f ff 84 <unknown>

ldff1sh { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ldff1sh { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xbf,0xdf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf df c4 <unknown>

ldff1sh { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-INST: ldff1sh { z23.d }, p3/z, [x13, z8.d, lsl #1]
// CHECK-ENCODING: [0xb7,0xad,0xe8,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: b7 ad e8 c4 <unknown>

ldff1sh { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ldff1sh { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x35,0x95,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 35 95 c4 <unknown>

ldff1sh { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ldff1sh { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x35,0xd5,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 35 d5 c4 <unknown>

ldff1sh { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-INST: ldff1sh { z0.d }, p0/z, [x0, z0.d, uxtw #1]
// CHECK-ENCODING: [0x00,0x20,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 a0 c4 <unknown>

ldff1sh { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-INST: ldff1sh { z0.d }, p0/z, [x0, z0.d, sxtw #1]
// CHECK-ENCODING: [0x00,0x20,0xe0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 e0 c4 <unknown>

ldff1sh { z31.s }, p7/z, [z31.s, #62]
// CHECK-INST: ldff1sh { z31.s }, p7/z, [z31.s, #62]
// CHECK-ENCODING: [0xff,0xbf,0xbf,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf bf 84 <unknown>

ldff1sh { z0.s }, p0/z, [z0.s]
// CHECK-INST: ldff1sh { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 a0 84 <unknown>

ldff1sh { z31.d }, p7/z, [z31.d, #62]
// CHECK-INST: ldff1sh { z31.d }, p7/z, [z31.d, #62]
// CHECK-ENCODING: [0xff,0xbf,0xbf,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf bf c4 <unknown>

ldff1sh { z0.d }, p0/z, [z0.d]
// CHECK-INST: ldff1sh { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 a0 c4 <unknown>
