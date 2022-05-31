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

ldnf1sb   z0.h, p0/z, [x0]
// CHECK-INST: ldnf1sb   { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xd0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 d0 a5 <unknown>

ldnf1sb   z0.s, p0/z, [x0]
// CHECK-INST: ldnf1sb   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xb0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 b0 a5 <unknown>

ldnf1sb   z0.d, p0/z, [x0]
// CHECK-INST: ldnf1sb   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x90,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 90 a5 <unknown>

ldnf1sb   { z0.h }, p0/z, [x0]
// CHECK-INST: ldnf1sb   { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xd0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 d0 a5 <unknown>

ldnf1sb   { z0.s }, p0/z, [x0]
// CHECK-INST: ldnf1sb   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xb0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 b0 a5 <unknown>

ldnf1sb   { z0.d }, p0/z, [x0]
// CHECK-INST: ldnf1sb   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x90,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 90 a5 <unknown>

ldnf1sb   { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1sb   { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xdf,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf df a5 <unknown>

ldnf1sb   { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1sb   { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xd5,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 d5 a5 <unknown>

ldnf1sb   { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1sb   { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xbf,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf bf a5 <unknown>

ldnf1sb   { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1sb   { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xb5,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 b5 a5 <unknown>

ldnf1sb   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1sb   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x9f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf 9f a5 <unknown>

ldnf1sb   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1sb   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x95,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 95 a5 <unknown>
