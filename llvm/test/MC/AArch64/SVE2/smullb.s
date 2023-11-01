// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN


smullb z0.h, z1.b, z2.b
// CHECK-INST: smullb z0.h, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x70,0x42,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 45427020 <unknown>

smullb z29.s, z30.h, z31.h
// CHECK-INST: smullb z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0x73,0x9f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 459f73dd <unknown>

smullb z31.d, z31.s, z31.s
// CHECK-INST: smullb z31.d, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x73,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 45df73ff <unknown>

smullb z0.s, z1.h, z7.h[7]
// CHECK-INST: smullb	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0xc8,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44bfc820 <unknown>

smullb z0.d, z1.s, z15.s[1]
// CHECK-INST: smullb	z0.d, z1.s, z15.s[1]
// CHECK-ENCODING: [0x20,0xc8,0xef,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44efc820 <unknown>
