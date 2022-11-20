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

ldnf1h     z0.h, p0/z, [x0]
// CHECK-INST: ldnf1h     { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xb0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4b0a000 <unknown>

ldnf1h     z0.s, p0/z, [x0]
// CHECK-INST: ldnf1h     { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xd0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4d0a000 <unknown>

ldnf1h     z0.d, p0/z, [x0]
// CHECK-INST: ldnf1h     { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xf0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4f0a000 <unknown>

ldnf1h    { z0.h }, p0/z, [x0]
// CHECK-INST: ldnf1h    { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xb0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4b0a000 <unknown>

ldnf1h    { z0.s }, p0/z, [x0]
// CHECK-INST: ldnf1h    { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xd0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4d0a000 <unknown>

ldnf1h    { z0.d }, p0/z, [x0]
// CHECK-INST: ldnf1h    { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0xf0,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4f0a000 <unknown>

ldnf1h    { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1h    { z31.h }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xbf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4bfbfff <unknown>

ldnf1h    { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1h    { z21.h }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xb5,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4b5b555 <unknown>

ldnf1h    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1h    { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xdf,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4dfbfff <unknown>

ldnf1h    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1h    { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xd5,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4d5b555 <unknown>

ldnf1h    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1h    { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0xff,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4ffbfff <unknown>

ldnf1h    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1h    { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0xf5,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4f5b555 <unknown>
