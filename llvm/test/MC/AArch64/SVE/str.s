// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

str     z0, [x0]
// CHECK-INST: str     z0, [x0]
// CHECK-ENCODING: [0x00,0x40,0x80,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5804000 <unknown>

str     z21, [x10, #-256, mul vl]
// CHECK-INST: str     z21, [x10, #-256, mul vl]
// CHECK-ENCODING: [0x55,0x41,0xa0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5a04155 <unknown>

str     z31, [sp, #255, mul vl]
// CHECK-INST: str     z31, [sp, #255, mul vl]
// CHECK-ENCODING: [0xff,0x5f,0x9f,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e59f5fff <unknown>

str     p0, [x0]
// CHECK-INST: str     p0, [x0]
// CHECK-ENCODING: [0x00,0x00,0x80,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5800000 <unknown>

str     p15, [sp, #-256, mul vl]
// CHECK-INST: str     p15, [sp, #-256, mul vl]
// CHECK-ENCODING: [0xef,0x03,0xa0,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5a003ef <unknown>

str     p5, [x10, #255, mul vl]
// CHECK-INST: str     p5, [x10, #255, mul vl]
// CHECK-ENCODING: [0x45,0x1d,0x9f,0xe5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e59f1d45 <unknown>
