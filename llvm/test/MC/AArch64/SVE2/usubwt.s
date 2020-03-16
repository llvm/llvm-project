// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


usubwt z0.h, z1.h, z2.b
// CHECK-INST: usubwt z0.h, z1.h, z2.b
// CHECK-ENCODING: [0x20,0x5c,0x42,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 5c 42 45 <unknown>

usubwt z29.s, z30.s, z31.h
// CHECK-INST: usubwt z29.s, z30.s, z31.h
// CHECK-ENCODING: [0xdd,0x5f,0x9f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: dd 5f 9f 45 <unknown>

usubwt z31.d, z31.d, z31.s
// CHECK-INST: usubwt z31.d, z31.d, z31.s
// CHECK-ENCODING: [0xff,0x5f,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 5f df 45 <unknown>
