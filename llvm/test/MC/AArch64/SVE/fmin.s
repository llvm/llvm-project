// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fmin    z0.h, p7/m, z0.h, z31.h
// CHECK-INST: fmin	z0.h, p7/m, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x47,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f 47 65 <unknown>

fmin    z0.s, p7/m, z0.s, z31.s
// CHECK-INST: fmin	z0.s, p7/m, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0x87,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f 87 65 <unknown>

fmin    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fmin	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc7,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 9f c7 65 <unknown>
