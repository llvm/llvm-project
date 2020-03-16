// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fcvtzu  z0.h, p0/m, z0.h
// CHECK-INST: fcvtzu  z0.h, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x5b,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 5b 65 <unknown>

fcvtzu  z0.s, p0/m, z0.h
// CHECK-INST: fcvtzu  z0.s, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x5d,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 5d 65 <unknown>

fcvtzu  z0.s, p0/m, z0.s
// CHECK-INST: fcvtzu  z0.s, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x9d,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 9d 65 <unknown>

fcvtzu  z0.s, p0/m, z0.d
// CHECK-INST: fcvtzu  z0.s, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd9,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 d9 65 <unknown>

fcvtzu  z0.d, p0/m, z0.h
// CHECK-INST: fcvtzu  z0.d, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x5f,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 5f 65 <unknown>

fcvtzu  z0.d, p0/m, z0.s
// CHECK-INST: fcvtzu  z0.d, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0xdd,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 dd 65 <unknown>

fcvtzu  z0.d, p0/m, z0.d
// CHECK-INST: fcvtzu  z0.d, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xdf,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 df 65 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z5.d, p0/z, z7.d
// CHECK-INST: movprfx	z5.d, p0/z, z7.d
// CHECK-ENCODING: [0xe5,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 20 d0 04 <unknown>

fcvtzu  z5.d, p0/m, z0.d
// CHECK-INST: fcvtzu	z5.d, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0xdf,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 a0 df 65 <unknown>

movprfx z5, z7
// CHECK-INST: movprfx	z5, z7
// CHECK-ENCODING: [0xe5,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e5 bc 20 04 <unknown>

fcvtzu  z5.d, p0/m, z0.d
// CHECK-INST: fcvtzu	z5.d, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0xdf,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 a0 df 65 <unknown>
