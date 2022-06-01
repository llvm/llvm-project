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


ssubwt z0.h, z1.h, z2.b
// CHECK-INST: ssubwt z0.h, z1.h, z2.b
// CHECK-ENCODING: [0x20,0x54,0x42,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 54 42 45 <unknown>

ssubwt z29.s, z30.s, z31.h
// CHECK-INST: ssubwt z29.s, z30.s, z31.h
// CHECK-ENCODING: [0xdd,0x57,0x9f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd 57 9f 45 <unknown>

ssubwt z31.d, z31.d, z31.s
// CHECK-INST: ssubwt z31.d, z31.d, z31.s
// CHECK-ENCODING: [0xff,0x57,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff 57 df 45 <unknown>
