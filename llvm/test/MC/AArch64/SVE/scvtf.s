// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

scvtf   z0.h, p0/m, z0.h
// CHECK-INST: scvtf   z0.h, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x52,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 6552a000 <unknown>

scvtf   z0.h, p0/m, z0.s
// CHECK-INST: scvtf   z0.h, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x54,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 6554a000 <unknown>

scvtf   z0.h, p0/m, z0.d
// CHECK-INST: scvtf   z0.h, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0x56,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 6556a000 <unknown>

scvtf   z0.s, p0/m, z0.s
// CHECK-INST: scvtf   z0.s, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x94,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 6594a000 <unknown>

scvtf   z0.s, p0/m, z0.d
// CHECK-INST: scvtf   z0.s, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd4,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d4a000 <unknown>

scvtf   z0.d, p0/m, z0.s
// CHECK-INST: scvtf   z0.d, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0xd0,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d0a000 <unknown>

scvtf   z0.d, p0/m, z0.d
// CHECK-INST: scvtf   z0.d, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd6,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d6a000 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z5.d, p0/z, z7.d
// CHECK-INST: movprfx	z5.d, p0/z, z7.d
// CHECK-ENCODING: [0xe5,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020e5 <unknown>

scvtf   z5.d, p0/m, z0.d
// CHECK-INST: scvtf	z5.d, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0xd6,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d6a005 <unknown>

movprfx z5, z7
// CHECK-INST: movprfx	z5, z7
// CHECK-ENCODING: [0xe5,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce5 <unknown>

scvtf   z5.d, p0/m, z0.d
// CHECK-INST: scvtf	z5.d, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0xd6,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d6a005 <unknown>
