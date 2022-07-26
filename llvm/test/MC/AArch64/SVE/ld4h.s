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

ld4h    { z0.h, z1.h, z2.h, z3.h }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ld4h    { z0.h, z1.h, z2.h, z3.h }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0xc0,0xe0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4e0c000 <unknown>

ld4h    { z5.h, z6.h, z7.h, z8.h }, p3/z, [x17, x16, lsl #1]
// CHECK-INST: ld4h    { z5.h, z6.h, z7.h, z8.h }, p3/z, [x17, x16, lsl #1]
// CHECK-ENCODING: [0x25,0xce,0xf0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4f0ce25 <unknown>

ld4h    { z0.h, z1.h, z2.h, z3.h }, p0/z, [x0]
// CHECK-INST: ld4h    { z0.h, z1.h, z2.h, z3.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xe0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4e0e000 <unknown>

ld4h    { z23.h, z24.h, z25.h, z26.h }, p3/z, [x13, #-32, mul vl]
// CHECK-INST: ld4h    { z23.h, z24.h, z25.h, z26.h }, p3/z, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0xe8,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4e8edb7 <unknown>

ld4h    { z21.h, z22.h, z23.h, z24.h }, p5/z, [x10, #20, mul vl]
// CHECK-INST: ld4h    { z21.h, z22.h, z23.h, z24.h }, p5/z, [x10, #20, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xe5,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4e5f555 <unknown>
