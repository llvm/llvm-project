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

ldff1b  { z31.b }, p7/z, [sp]
// CHECK-INST: ldff1b  { z31.b }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x1f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 1f a4 <unknown>

ldff1b  { z31.h }, p7/z, [sp]
// CHECK-INST: ldff1b  { z31.h }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x3f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 3f a4 <unknown>

ldff1b  { z31.s }, p7/z, [sp]
// CHECK-INST: ldff1b  { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x5f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 5f a4 <unknown>

ldff1b  { z31.d }, p7/z, [sp]
// CHECK-INST: ldff1b  { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x7f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 7f a4 <unknown>

ldff1b  { z31.b }, p7/z, [sp, xzr]
// CHECK-INST: ldff1b  { z31.b }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x1f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 1f a4 <unknown>

ldff1b  { z31.h }, p7/z, [sp, xzr]
// CHECK-INST: ldff1b  { z31.h }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x3f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 3f a4 <unknown>

ldff1b  { z31.s }, p7/z, [sp, xzr]
// CHECK-INST: ldff1b  { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x5f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 5f a4 <unknown>

ldff1b  { z31.d }, p7/z, [sp, xzr]
// CHECK-INST: ldff1b  { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x7f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f 7f a4 <unknown>

ldff1b  { z0.h }, p0/z, [x0, x0]
// CHECK-INST: ldff1b  { z0.h }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x60,0x20,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 20 a4 <unknown>

ldff1b  { z0.s }, p0/z, [x0, x0]
// CHECK-INST: ldff1b  { z0.s }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x60,0x40,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 40 a4 <unknown>

ldff1b  { z0.d }, p0/z, [x0, x0]
// CHECK-INST: ldff1b  { z0.d }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x60,0x60,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 60 a4 <unknown>

ldff1b    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ldff1b    { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x60,0x00,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 00 84 <unknown>

ldff1b    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ldff1b    { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x60,0x40,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 40 84 <unknown>

ldff1b  { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ldff1b  { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xff,0x5f,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff ff 5f c4 <unknown>

ldff1b  { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ldff1b  { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x75,0x15,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 75 15 c4 <unknown>

ldff1b  { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ldff1b  { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x75,0x55,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 75 55 c4 <unknown>

ldff1b  { z31.s }, p7/z, [z31.s, #31]
// CHECK-INST: ldff1b  { z31.s }, p7/z, [z31.s, #31]
// CHECK-ENCODING: [0xff,0xff,0x3f,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff ff 3f 84 <unknown>

ldff1b  { z0.s }, p0/z, [z0.s]
// CHECK-INST: ldff1b  { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0xe0,0x20,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 20 84 <unknown>

ldff1b  { z31.d }, p7/z, [z31.d, #31]
// CHECK-INST: ldff1b  { z31.d }, p7/z, [z31.d, #31]
// CHECK-ENCODING: [0xff,0xff,0x3f,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff ff 3f c4 <unknown>

ldff1b  { z0.d }, p0/z, [z0.d]
// CHECK-INST: ldff1b  { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xe0,0x20,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 e0 20 c4 <unknown>
