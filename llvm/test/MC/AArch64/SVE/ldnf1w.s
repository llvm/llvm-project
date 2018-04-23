// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ldnf1w     z0.s, p0/z, [x0]
// CHECK-INST: ldnf1w     { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x50,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 50 a5 <unknown>

ldnf1w     z0.d, p0/z, [x0]
// CHECK-INST: ldnf1w     { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x70,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 70 a5 <unknown>

ldnf1w    { z0.s }, p0/z, [x0]
// CHECK-INST: ldnf1w    { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x50,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 50 a5 <unknown>

ldnf1w    { z0.d }, p0/z, [x0]
// CHECK-INST: ldnf1w    { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x70,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 70 a5 <unknown>

ldnf1w    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1w    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x5f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 5f a5 <unknown>

ldnf1w    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1w    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x55,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 55 a5 <unknown>

ldnf1w    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1w    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x7f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 7f a5 <unknown>

ldnf1w    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1w    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x75,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 75 a5 <unknown>
