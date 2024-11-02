// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

scvtf   z0.h, p0/m, z0.h
// CHECK-INST: scvtf   z0.h, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x52,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 52 65 <unknown>

scvtf   z0.h, p0/m, z0.s
// CHECK-INST: scvtf   z0.h, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x54,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 54 65 <unknown>

scvtf   z0.h, p0/m, z0.d
// CHECK-INST: scvtf   z0.h, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0x56,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 56 65 <unknown>

scvtf   z0.s, p0/m, z0.s
// CHECK-INST: scvtf   z0.s, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x94,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 94 65 <unknown>

scvtf   z0.s, p0/m, z0.d
// CHECK-INST: scvtf   z0.s, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd4,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 d4 65 <unknown>

scvtf   z0.d, p0/m, z0.s
// CHECK-INST: scvtf   z0.d, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0xd0,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 d0 65 <unknown>

scvtf   z0.d, p0/m, z0.d
// CHECK-INST: scvtf   z0.d, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd6,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 d6 65 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z5.d, p0/z, z7.d
// CHECK-INST: movprfx	z5.d, p0/z, z7.d
// CHECK-ENCODING: [0xe5,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5 20 d0 04 <unknown>

scvtf   z5.d, p0/m, z0.d
// CHECK-INST: scvtf	z5.d, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0xd6,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05 a0 d6 65 <unknown>

movprfx z5, z7
// CHECK-INST: movprfx	z5, z7
// CHECK-ENCODING: [0xe5,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5 bc 20 04 <unknown>

scvtf   z5.d, p0/m, z0.d
// CHECK-INST: scvtf	z5.d, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0xd6,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05 a0 d6 65 <unknown>
