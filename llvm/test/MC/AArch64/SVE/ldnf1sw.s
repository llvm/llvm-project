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

ldnf1sw   z0.d, p0/z, [x0]
// CHECK-INST: ldnf1sw   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x90,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a490a000 <unknown>

ldnf1sw   { z0.d }, p0/z, [x0]
// CHECK-INST: ldnf1sw   { z0.d }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xa0,0x90,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a490a000 <unknown>

ldnf1sw   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-INST: ldnf1sw   { z31.d }, p7/z, [sp, #-1, mul vl]
// CHECK-ENCODING: [0xff,0xbf,0x9f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a49fbfff <unknown>

ldnf1sw   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-INST: ldnf1sw   { z21.d }, p5/z, [x10, #5, mul vl]
// CHECK-ENCODING: [0x55,0xb5,0x95,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a495b555 <unknown>
