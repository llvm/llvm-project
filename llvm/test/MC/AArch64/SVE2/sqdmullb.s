// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


sqdmullb z0.h, z1.b, z2.b
// CHECK-INST: sqdmullb z0.h, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x60,0x42,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 60 42 45 <unknown>

sqdmullb z29.s, z30.h, z31.h
// CHECK-INST: sqdmullb z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0x63,0x9f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: dd 63 9f 45 <unknown>

sqdmullb z31.d, z31.s, z31.s
// CHECK-INST: sqdmullb z31.d, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x63,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 63 df 45 <unknown>

sqdmullb z0.s, z1.h, z7.h[7]
// CHECK-INST: sqdmullb	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0xe8,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 e8 bf 44 <unknown>

sqdmullb z0.d, z1.s, z15.s[1]
// CHECK-INST: sqdmullb	z0.d, z1.s, z15.s[1]
// CHECK-ENCODING: [0x20,0xe8,0xef,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 e8 ef 44 <unknown>
