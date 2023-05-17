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

ucvtf   z0.h, p0/m, z0.h
// CHECK-INST: ucvtf   z0.h, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x53,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 6553a000 <unknown>

ucvtf   z0.h, p0/m, z0.s
// CHECK-INST: ucvtf   z0.h, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x55,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 6555a000 <unknown>

ucvtf   z0.h, p0/m, z0.d
// CHECK-INST: ucvtf   z0.h, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0x57,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 6557a000 <unknown>

ucvtf   z0.s, p0/m, z0.s
// CHECK-INST: ucvtf   z0.s, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x95,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 6595a000 <unknown>

ucvtf   z0.s, p0/m, z0.d
// CHECK-INST: ucvtf   z0.s, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd5,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d5a000 <unknown>

ucvtf   z0.d, p0/m, z0.s
// CHECK-INST: ucvtf   z0.d, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0xd1,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d1a000 <unknown>

ucvtf   z0.d, p0/m, z0.d
// CHECK-INST: ucvtf   z0.d, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd7,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d7a000 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z5.d, p0/z, z7.d
// CHECK-INST: movprfx	z5.d, p0/z, z7.d
// CHECK-ENCODING: [0xe5,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020e5 <unknown>

ucvtf   z5.d, p0/m, z0.d
// CHECK-INST: ucvtf	z5.d, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0xd7,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d7a005 <unknown>

movprfx z5, z7
// CHECK-INST: movprfx	z5, z7
// CHECK-ENCODING: [0xe5,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce5 <unknown>

ucvtf   z5.d, p0/m, z0.d
// CHECK-INST: ucvtf	z5.d, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0xd7,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65d7a005 <unknown>
