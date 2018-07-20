// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ldnf1d     z0.d, p0/z, [x0]
// CHECK-INST: ldnf1d     { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xf0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 f0 a5 <unknown>

ldnf1d    { z0.d }, p0/z, [x0]
// CHECK-INST: ldnf1d    { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xf0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 f0 a5 <unknown>

ldnf1d    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1d    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xff,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf ff a5 <unknown>

ldnf1d    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1d    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xf5,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 f5 a5 <unknown>
