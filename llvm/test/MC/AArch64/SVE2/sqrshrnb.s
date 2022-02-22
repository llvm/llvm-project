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

sqrshrnb     z0.b, z0.h, #1
// CHECK-INST: sqrshrnb	z0.b, z0.h, #1
// CHECK-ENCODING: [0x00,0x28,0x2f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 28 2f 45 <unknown>

sqrshrnb     z31.b, z31.h, #8
// CHECK-INST: sqrshrnb	z31.b, z31.h, #8
// CHECK-ENCODING: [0xff,0x2b,0x28,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2b 28 45 <unknown>

sqrshrnb     z0.h, z0.s, #1
// CHECK-INST: sqrshrnb	z0.h, z0.s, #1
// CHECK-ENCODING: [0x00,0x28,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 28 3f 45 <unknown>

sqrshrnb     z31.h, z31.s, #16
// CHECK-INST: sqrshrnb	z31.h, z31.s, #16
// CHECK-ENCODING: [0xff,0x2b,0x30,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2b 30 45 <unknown>

sqrshrnb     z0.s, z0.d, #1
// CHECK-INST: sqrshrnb	z0.s, z0.d, #1
// CHECK-ENCODING: [0x00,0x28,0x7f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 28 7f 45 <unknown>

sqrshrnb     z31.s, z31.d, #32
// CHECK-INST: sqrshrnb	z31.s, z31.d, #32
// CHECK-ENCODING: [0xff,0x2b,0x60,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 2b 60 45 <unknown>
