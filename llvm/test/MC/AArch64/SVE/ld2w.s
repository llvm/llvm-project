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

ld2w    { z0.s, z1.s }, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ld2w    { z0.s, z1.s }, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0xc0,0x20,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a520c000 <unknown>

ld2w    { z5.s, z6.s }, p3/z, [x17, x16, lsl #2]
// CHECK-INST: ld2w    { z5.s, z6.s }, p3/z, [x17, x16, lsl #2]
// CHECK-ENCODING: [0x25,0xce,0x30,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a530ce25 <unknown>

ld2w    { z0.s, z1.s }, p0/z, [x0]
// CHECK-INST: ld2w    { z0.s, z1.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0x20,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a520e000 <unknown>

ld2w    { z23.s, z24.s }, p3/z, [x13, #-16, mul vl]
// CHECK-INST: ld2w    { z23.s, z24.s }, p3/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0x28,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a528edb7 <unknown>

ld2w    { z21.s, z22.s }, p5/z, [x10, #10, mul vl]
// CHECK-INST: ld2w    { z21.s, z22.s }, p5/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0x25,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a525f555 <unknown>
