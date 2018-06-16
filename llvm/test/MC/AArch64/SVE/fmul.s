// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fmul    z0.h, p0/m, z0.h, #0.5000000000000
// CHECK-INST: fmul    z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x5a,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 5a 65 <unknown>

fmul    z0.h, p0/m, z0.h, #0.5
// CHECK-INST: fmul    z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x5a,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 5a 65 <unknown>

fmul    z0.s, p0/m, z0.s, #0.5
// CHECK-INST: fmul    z0.s, p0/m, z0.s, #0.5
// CHECK-ENCODING: [0x00,0x80,0x9a,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 9a 65 <unknown>

fmul    z0.d, p0/m, z0.d, #0.5
// CHECK-INST: fmul    z0.d, p0/m, z0.d, #0.5
// CHECK-ENCODING: [0x00,0x80,0xda,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 da 65 <unknown>

fmul    z31.h, p7/m, z31.h, #2.0
// CHECK-INST: fmul    z31.h, p7/m, z31.h, #2.0
// CHECK-ENCODING: [0x3f,0x9c,0x5a,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c 5a 65 <unknown>

fmul    z31.s, p7/m, z31.s, #2.0
// CHECK-INST: fmul    z31.s, p7/m, z31.s, #2.0
// CHECK-ENCODING: [0x3f,0x9c,0x9a,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c 9a 65 <unknown>

fmul    z31.d, p7/m, z31.d, #2.0
// CHECK-INST: fmul    z31.d, p7/m, z31.d, #2.0
// CHECK-ENCODING: [0x3f,0x9c,0xda,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 3f 9c da 65 <unknown>
