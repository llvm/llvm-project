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

smulh z0.b, z1.b, z2.b
// CHECK-INST: smulh z0.b, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x68,0x22,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 68 22 04 <unknown>

smulh z0.h, z1.h, z2.h
// CHECK-INST: smulh z0.h, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x68,0x62,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 68 62 04 <unknown>

smulh z29.s, z30.s, z31.s
// CHECK-INST: smulh z29.s, z30.s, z31.s
// CHECK-ENCODING: [0xdd,0x6b,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 6b bf 04 <unknown>

smulh z31.d, z31.d, z31.d
// CHECK-INST: smulh z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x6b,0xff,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 6b ff 04 <unknown>
