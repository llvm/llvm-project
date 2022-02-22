// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


fcvtx    z0.s, p0/m, z0.d
// CHECK-INST: fcvtx    z0.s, p0/m, z0.d
// CHECK-ENCODING: [0x00,0xa0,0x0a,0x65]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 a0 0a 65 <unknown>

fcvtx    z30.s, p7/m, z31.d
// CHECK-INST: fcvtx    z30.s, p7/m, z31.d
// CHECK-ENCODING: [0xfe,0xbf,0x0a,0x65]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: fe bf 0a 65 <unknown>



// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z5.d, p0/z, z7.d
// CHECK-INST: movprfx	z5.d, p0/z, z7.d
// CHECK-ENCODING: [0xe5,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5 20 d0 04 <unknown>

fcvtx    z5.s, p0/m, z0.d
// CHECK-INST: fcvtx	z5.s, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0x0a,0x65]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 05 a0 0a 65 <unknown>

movprfx z5, z7
// CHECK-INST: movprfx	z5, z7
// CHECK-ENCODING: [0xe5,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e5 bc 20 04 <unknown>

fcvtx    z5.s, p0/m, z0.d
// CHECK-INST: fcvtx	z5.s, p0/m, z0.d
// CHECK-ENCODING: [0x05,0xa0,0x0a,0x65]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 05 a0 0a 65 <unknown>
