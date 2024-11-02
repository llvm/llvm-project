// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld1rqw  { z0.s }, p0/z, [x0]
// CHECK-INST: ld1rqw  { z0.s }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0x20,0x00,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 20 00 a5 <unknown>

ld1rqw  { z0.s }, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ld1rqw  { z0.s }, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x00,0x00,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 00 a5 <unknown>

ld1rqw  { z31.s }, p7/z, [sp, #-16]
// CHECK-INST: ld1rqw  { z31.s }, p7/z, [sp, #-16]
// CHECK-ENCODING: [0xff,0x3f,0x0f,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 3f 0f a5 <unknown>

ld1rqw  { z23.s }, p3/z, [x13, #-128]
// CHECK-INST: ld1rqw  { z23.s }, p3/z, [x13, #-128]
// CHECK-ENCODING: [0xb7,0x2d,0x08,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 2d 08 a5 <unknown>

ld1rqw  { z23.s }, p3/z, [x13, #112]
// CHECK-INST: ld1rqw  { z23.s }, p3/z, [x13, #112]
// CHECK-ENCODING: [0xb7,0x2d,0x07,0xa5]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 2d 07 a5 <unknown>
