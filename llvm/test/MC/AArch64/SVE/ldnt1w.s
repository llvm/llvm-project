// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ldnt1w  z0.s, p0/z, [x0]
// CHECK-INST: ldnt1w  { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x00,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a500e000 <unknown>

ldnt1w  { z0.s }, p0/z, [x0]
// CHECK-INST: ldnt1w  { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x00,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a500e000 <unknown>

ldnt1w  { z23.s }, p3/z, [x13, #-8, mul vl]
// CHECK-INST: ldnt1w  { z23.s }, p3/z, [x13, #-8, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x08,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a508edb7 <unknown>

ldnt1w  { z21.s }, p5/z, [x10, #7, mul vl]
// CHECK-INST: ldnt1w  { z21.s }, p5/z, [x10, #7, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x07,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a507f555 <unknown>

ldnt1w  { z0.s }, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ldnt1w  { z0.s }, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0xc0,0x00,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a500c000 <unknown>
