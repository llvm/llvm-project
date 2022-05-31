// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

mul z0.b, z1.b, z2.b
// CHECK-INST: mul z0.b, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x60,0x22,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 60 22 04 <unknown>

mul z0.h, z1.h, z2.h
// CHECK-INST: mul z0.h, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x60,0x62,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 60 62 04 <unknown>

mul z29.s, z30.s, z31.s
// CHECK-INST: mul z29.s, z30.s, z31.s
// CHECK-ENCODING: [0xdd,0x63,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 63 bf 04 <unknown>

mul z31.d, z31.d, z31.d
// CHECK-INST: mul z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x63,0xff,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 63 ff 04 <unknown>

mul z0.h, z1.h, z7.h[7]
// CHECK-INST: mul	z0.h, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0xf8,0x7f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 f8 7f 44 <unknown>

mul z0.s, z1.s, z7.s[3]
// CHECK-INST: mul	z0.s, z1.s, z7.s[3]
// CHECK-ENCODING: [0x20,0xf8,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 f8 bf 44 <unknown>

mul z0.d, z1.d, z15.d[1]
// CHECK-INST: mul	z0.d, z1.d, z15.d[1]
// CHECK-ENCODING: [0x20,0xf8,0xff,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 f8 ff 44 <unknown>
