// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fcvtzs  z0.h, p0/m, z0.h
// CHECK-INST: fcvtzs  z0.h, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x5a,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 5a 65 <unknown>

fcvtzs  z0.s, p0/m, z0.h
// CHECK-INST: fcvtzs  z0.s, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x5c,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 5c 65 <unknown>

fcvtzs  z0.s, p0/m, z0.s
// CHECK-INST: fcvtzs  z0.s, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x9c,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 9c 65 <unknown>

fcvtzs  z0.s, p0/m, z0.d
// CHECK-INST: fcvtzs  z0.s, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd8,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 d8 65 <unknown>

fcvtzs  z0.d, p0/m, z0.h
// CHECK-INST: fcvtzs  z0.d, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x5e,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 5e 65 <unknown>

fcvtzs  z0.d, p0/m, z0.s
// CHECK-INST: fcvtzs  z0.d, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0xdc,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 dc 65 <unknown>

fcvtzs  z0.d, p0/m, z0.d
// CHECK-INST: fcvtzs  z0.d, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xde,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 de 65 <unknown>
