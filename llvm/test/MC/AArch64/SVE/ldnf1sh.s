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

ldnf1sh   z0.s, p0/z, [x0]
// CHECK-INST: ldnf1sh   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x30,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a530a000 <unknown>

ldnf1sh   z0.d, p0/z, [x0]
// CHECK-INST: ldnf1sh   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x10,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a510a000 <unknown>

ldnf1sh   { z0.s }, p0/z, [x0]
// CHECK-INST: ldnf1sh   { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x30,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a530a000 <unknown>

ldnf1sh   { z0.d }, p0/z, [x0]
// CHECK-INST: ldnf1sh   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x10,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a510a000 <unknown>

ldnf1sh   { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1sh   { z31.s }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x3f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a53fbfff <unknown>

ldnf1sh   { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1sh   { z21.s }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x35,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a535b555 <unknown>

ldnf1sh   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1sh   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x1f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a51fbfff <unknown>

ldnf1sh   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1sh   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x15,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a515b555 <unknown>
