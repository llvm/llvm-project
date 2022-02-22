// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sqshrunt     z0.b, z0.h, #1
// CHECK-INST: sqshrunt	z0.b, z0.h, #1
// CHECK-ENCODING: [0x00,0x04,0x2f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 04 2f 45 <unknown>

sqshrunt     z31.b, z31.h, #8
// CHECK-INST: sqshrunt	z31.b, z31.h, #8
// CHECK-ENCODING: [0xff,0x07,0x28,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 07 28 45 <unknown>

sqshrunt     z0.h, z0.s, #1
// CHECK-INST: sqshrunt	z0.h, z0.s, #1
// CHECK-ENCODING: [0x00,0x04,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 04 3f 45 <unknown>

sqshrunt     z31.h, z31.s, #16
// CHECK-INST: sqshrunt	z31.h, z31.s, #16
// CHECK-ENCODING: [0xff,0x07,0x30,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 07 30 45 <unknown>

sqshrunt     z0.s, z0.d, #1
// CHECK-INST: sqshrunt	z0.s, z0.d, #1
// CHECK-ENCODING: [0x00,0x04,0x7f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 04 7f 45 <unknown>

sqshrunt     z31.s, z31.d, #32
// CHECK-INST: sqshrunt	z31.s, z31.d, #32
// CHECK-ENCODING: [0xff,0x07,0x60,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 07 60 45 <unknown>
