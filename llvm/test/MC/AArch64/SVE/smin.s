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

smin    z0.b, z0.b, #-128
// CHECK-INST: smin	z0.b, z0.b, #-128
// CHECK-ENCODING: [0x00,0xd0,0x2a,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d0 2a 25 <unknown>

smin    z31.b, z31.b, #127
// CHECK-INST: smin	z31.b, z31.b, #127
// CHECK-ENCODING: [0xff,0xcf,0x2a,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff cf 2a 25 <unknown>

smin    z0.h, z0.h, #-128
// CHECK-INST: smin	z0.h, z0.h, #-128
// CHECK-ENCODING: [0x00,0xd0,0x6a,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d0 6a 25 <unknown>

smin    z31.h, z31.h, #127
// CHECK-INST: smin	z31.h, z31.h, #127
// CHECK-ENCODING: [0xff,0xcf,0x6a,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff cf 6a 25 <unknown>

smin    z0.s, z0.s, #-128
// CHECK-INST: smin	z0.s, z0.s, #-128
// CHECK-ENCODING: [0x00,0xd0,0xaa,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d0 aa 25 <unknown>

smin    z31.s, z31.s, #127
// CHECK-INST: smin	z31.s, z31.s, #127
// CHECK-ENCODING: [0xff,0xcf,0xaa,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff cf aa 25 <unknown>

smin    z0.d, z0.d, #-128
// CHECK-INST: smin	z0.d, z0.d, #-128
// CHECK-ENCODING: [0x00,0xd0,0xea,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 d0 ea 25 <unknown>

smin    z31.d, z31.d, #127
// CHECK-INST: smin	z31.d, z31.d, #127
// CHECK-ENCODING: [0xff,0xcf,0xea,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff cf ea 25 <unknown>

smin    z31.b, p7/m, z31.b, z31.b
// CHECK-INST: smin	z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x1f,0x0a,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 0a 04 <unknown>

smin    z31.h, p7/m, z31.h, z31.h
// CHECK-INST: smin	z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x4a,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 4a 04 <unknown>

smin    z31.s, p7/m, z31.s, z31.s
// CHECK-INST: smin	z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0x8a,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 8a 04 <unknown>

smin    z31.d, p7/m, z31.d, z31.d
// CHECK-INST: smin	z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x1f,0xca,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f ca 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 3c d0 04 <unknown>

smin    z4.d, p7/m, z4.d, z31.d
// CHECK-INST: smin	z4.d, p7/m, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x1f,0xca,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 1f ca 04 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

smin    z4.d, p7/m, z4.d, z31.d
// CHECK-INST: smin	z4.d, p7/m, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x1f,0xca,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 1f ca 04 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

smin    z31.d, z31.d, #127
// CHECK-INST: smin	z31.d, z31.d, #127
// CHECK-ENCODING: [0xff,0xcf,0xea,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff cf ea 25 <unknown>
