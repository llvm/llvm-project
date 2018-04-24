// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ldff1h  { z31.h }, p7/z, [sp]
// CHECK-INST: ldff1h  { z31.h }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xbf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f bf a4 <unknown>

ldff1h  { z31.s }, p7/z, [sp]
// CHECK-INST: ldff1h  { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xdf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f df a4 <unknown>

ldff1h  { z31.d }, p7/z, [sp]
// CHECK-INST: ldff1h  { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xff,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f ff a4 <unknown>

ldff1h  { z31.h }, p7/z, [sp, xzr, lsl #1]
// CHECK-INST: ldff1h  { z31.h }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xbf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f bf a4 <unknown>

ldff1h  { z31.s }, p7/z, [sp, xzr, lsl #1]
// CHECK-INST: ldff1h  { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xdf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f df a4 <unknown>

ldff1h  { z31.d }, p7/z, [sp, xzr, lsl #1]
// CHECK-INST: ldff1h  { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xff,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 7f ff a4 <unknown>

ldff1h  { z0.h }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldff1h  { z0.h }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0xa0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 a0 a4 <unknown>

ldff1h  { z0.s }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldff1h  { z0.s }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0xc0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 c0 a4 <unknown>

ldff1h  { z0.d }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ldff1h  { z0.d }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x60,0xe0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 60 e0 a4 <unknown>
