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


smullt z0.h, z1.b, z2.b
// CHECK-INST: smullt z0.h, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x74,0x42,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 45427420 <unknown>

smullt z29.s, z30.h, z31.h
// CHECK-INST: smullt z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0x77,0x9f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 459f77dd <unknown>

smullt z31.d, z31.s, z31.s
// CHECK-INST: smullt z31.d, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x77,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 45df77ff <unknown>

smullt z0.s, z1.h, z7.h[7]
// CHECK-INST: smullt	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0xcc,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44bfcc20 <unknown>

smullt z0.d, z1.s, z15.s[1]
// CHECK-INST: smullt	z0.d, z1.s, z15.s[1]
// CHECK-ENCODING: [0x20,0xcc,0xef,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44efcc20 <unknown>
