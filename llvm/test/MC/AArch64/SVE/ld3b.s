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

ld3b    { z0.b, z1.b, z2.b }, p0/z, [x0, x0]
// CHECK-INST: ld3b    { z0.b - z2.b }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0xc0,0x40,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a440c000 <unknown>

ld3b    { z5.b, z6.b, z7.b }, p3/z, [x17, x16]
// CHECK-INST: ld3b    { z5.b - z7.b }, p3/z, [x17, x16]
// CHECK-ENCODING: [0x25,0xce,0x50,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a450ce25 <unknown>

ld3b    { z0.b, z1.b, z2.b }, p0/z, [x0]
// CHECK-INST: ld3b    { z0.b - z2.b }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x40,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a440e000 <unknown>

ld3b    { z23.b, z24.b, z25.b }, p3/z, [x13, #-24, mul vl]
// CHECK-INST: ld3b    { z23.b - z25.b }, p3/z, [x13, #-24, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x48,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a448edb7 <unknown>

ld3b    { z21.b, z22.b, z23.b }, p5/z, [x10, #15, mul vl]
// CHECK-INST: ld3b    { z21.b - z23.b }, p5/z, [x10, #15, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x45,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a445f555 <unknown>
