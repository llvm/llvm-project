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


umullt z0.h, z1.b, z2.b
// CHECK-INST: umullt z0.h, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x7c,0x42,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 7c 42 45 <unknown>

umullt z29.s, z30.h, z31.h
// CHECK-INST: umullt z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0x7f,0x9f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 7f 9f 45 <unknown>

umullt z31.d, z31.s, z31.s
// CHECK-INST: umullt z31.d, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x7f,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 7f df 45 <unknown>

umullt z0.s, z1.h, z7.h[7]
// CHECK-INST: umullt	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0xdc,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 dc bf 44 <unknown>

umullt z0.d, z1.s, z15.s[1]
// CHECK-INST: umullt	z0.d, z1.s, z15.s[1]
// CHECK-ENCODING: [0x20,0xdc,0xef,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 dc ef 44 <unknown>
