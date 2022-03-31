// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ucvtf   z0.h, p0/m, z0.h
// CHECK-INST: ucvtf   z0.h, p0/m, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x53,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 53 65 <unknown>

ucvtf   z0.h, p0/m, z0.s
// CHECK-INST: ucvtf   z0.h, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x55,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 55 65 <unknown>

ucvtf   z0.h, p0/m, z0.d
// CHECK-INST: ucvtf   z0.h, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0x57,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 57 65 <unknown>

ucvtf   z0.s, p0/m, z0.s
// CHECK-INST: ucvtf   z0.s, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x95,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 95 65 <unknown>

ucvtf   z0.s, p0/m, z0.d
// CHECK-INST: ucvtf   z0.s, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd5,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 d5 65 <unknown>

ucvtf   z0.d, p0/m, z0.s
// CHECK-INST: ucvtf   z0.d, p0/m, z0.s
// CHECK-ENCODING: [0x00,0xa0,0xd1,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 d1 65 <unknown>

ucvtf   z0.d, p0/m, z0.d
// CHECK-INST: ucvtf   z0.d, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd7,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 d7 65 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z5.d, p0/z, z7.d
// CHECK-INST: movprfx	z5.d, p0/z, z7.d
// CHECK-ENCODING: [0xe5,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5 20 d0 04 <unknown>

ucvtf   z5.d, p0/m, z0.d
// CHECK-INST: ucvtf	z5.d, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0xd7,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05 a0 d7 65 <unknown>

movprfx z5, z7
// CHECK-INST: movprfx	z5, z7
// CHECK-ENCODING: [0xe5,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5 bc 20 04 <unknown>

ucvtf   z5.d, p0/m, z0.d
// CHECK-INST: ucvtf	z5.d, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0xd7,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05 a0 d7 65 <unknown>
