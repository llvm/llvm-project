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

fcadd   z0.h, p0/m, z0.h, z0.h, #90
// CHECK-INST: fcadd   z0.h, p0/m, z0.h, z0.h, #90
// CHECK-ENCODING: [0x00,0x80,0x40,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 40 64 <unknown>

fcadd   z0.s, p0/m, z0.s, z0.s, #90
// CHECK-INST: fcadd   z0.s, p0/m, z0.s, z0.s, #90
// CHECK-ENCODING: [0x00,0x80,0x80,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 80 64 <unknown>

fcadd   z0.d, p0/m, z0.d, z0.d, #90
// CHECK-INST: fcadd   z0.d, p0/m, z0.d, z0.d, #90
// CHECK-ENCODING: [0x00,0x80,0xc0,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 c0 64 <unknown>

fcadd   z31.h, p7/m, z31.h, z31.h, #270
// CHECK-INST: fcadd   z31.h, p7/m, z31.h, z31.h, #270
// CHECK-ENCODING: [0xff,0x9f,0x41,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f 41 64 <unknown>

fcadd   z31.s, p7/m, z31.s, z31.s, #270
// CHECK-INST: fcadd   z31.s, p7/m, z31.s, z31.s, #270
// CHECK-ENCODING: [0xff,0x9f,0x81,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f 81 64 <unknown>

fcadd   z31.d, p7/m, z31.d, z31.d, #270
// CHECK-INST: fcadd   z31.d, p7/m, z31.d, z31.d, #270
// CHECK-ENCODING: [0xff,0x9f,0xc1,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 9f c1 64 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 3c d0 04 <unknown>

fcadd   z4.d, p7/m, z4.d, z31.d, #270
// CHECK-INST: fcadd	z4.d, p7/m, z4.d, z31.d, #270
// CHECK-ENCODING: [0xe4,0x9f,0xc1,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 9f c1 64 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

fcadd   z4.d, p7/m, z4.d, z31.d, #270
// CHECK-INST: fcadd	z4.d, p7/m, z4.d, z31.d, #270
// CHECK-ENCODING: [0xe4,0x9f,0xc1,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 9f c1 64 <unknown>
