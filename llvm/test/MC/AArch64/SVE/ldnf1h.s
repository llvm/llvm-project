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

ldnf1h     z0.h, p0/z, [x0]
// CHECK-INST: ldnf1h     { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xb0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 b0 a4 <unknown>

ldnf1h     z0.s, p0/z, [x0]
// CHECK-INST: ldnf1h     { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xd0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 d0 a4 <unknown>

ldnf1h     z0.d, p0/z, [x0]
// CHECK-INST: ldnf1h     { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xf0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 f0 a4 <unknown>

ldnf1h    { z0.h }, p0/z, [x0]
// CHECK-INST: ldnf1h    { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xb0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 b0 a4 <unknown>

ldnf1h    { z0.s }, p0/z, [x0]
// CHECK-INST: ldnf1h    { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xd0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 d0 a4 <unknown>

ldnf1h    { z0.d }, p0/z, [x0]
// CHECK-INST: ldnf1h    { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xf0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 f0 a4 <unknown>

ldnf1h    { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1h    { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xbf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf bf a4 <unknown>

ldnf1h    { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1h    { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xb5,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 b5 a4 <unknown>

ldnf1h    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1h    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xdf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf df a4 <unknown>

ldnf1h    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1h    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xd5,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 d5 a4 <unknown>

ldnf1h    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1h    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xff,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bf ff a4 <unknown>

ldnf1h    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1h    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xf5,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 55 b5 f5 a4 <unknown>
