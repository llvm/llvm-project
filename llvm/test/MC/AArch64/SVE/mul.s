// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

mul z0.b, p7/m, z0.b, z31.b
// CHECK-INST: mul	z0.b, p7/m, z0.b, z31.b
// CHECK-ENCODING: [0xe0,0x1f,0x10,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 1f 10 04 <unknown>

mul z0.h, p7/m, z0.h, z31.h
// CHECK-INST: mul	z0.h, p7/m, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x1f,0x50,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 1f 50 04 <unknown>

mul z0.s, p7/m, z0.s, z31.s
// CHECK-INST: mul	z0.s, p7/m, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x1f,0x90,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 1f 90 04 <unknown>

mul z0.d, p7/m, z0.d, z31.d
// CHECK-INST: mul	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x1f,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 1f d0 04 <unknown>

mul z31.b, z31.b, #-128
// CHECK-INST: mul	z31.b, z31.b, #-128
// CHECK-ENCODING: [0x1f,0xd0,0x30,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f d0 30 25 <unknown>

mul z31.b, z31.b, #127
// CHECK-INST: mul	z31.b, z31.b, #127
// CHECK-ENCODING: [0xff,0xcf,0x30,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff cf 30 25 <unknown>

mul z31.h, z31.h, #-128
// CHECK-INST: mul	z31.h, z31.h, #-128
// CHECK-ENCODING: [0x1f,0xd0,0x70,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f d0 70 25 <unknown>

mul z31.h, z31.h, #127
// CHECK-INST: mul	z31.h, z31.h, #127
// CHECK-ENCODING: [0xff,0xcf,0x70,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff cf 70 25 <unknown>

mul z31.s, z31.s, #-128
// CHECK-INST: mul	z31.s, z31.s, #-128
// CHECK-ENCODING: [0x1f,0xd0,0xb0,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f d0 b0 25 <unknown>

mul z31.s, z31.s, #127
// CHECK-INST: mul	z31.s, z31.s, #127
// CHECK-ENCODING: [0xff,0xcf,0xb0,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff cf b0 25 <unknown>

mul z31.d, z31.d, #-128
// CHECK-INST: mul	z31.d, z31.d, #-128
// CHECK-ENCODING: [0x1f,0xd0,0xf0,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 1f d0 f0 25 <unknown>

mul z31.d, z31.d, #127
// CHECK-INST: mul	z31.d, z31.d, #127
// CHECK-ENCODING: [0xff,0xcf,0xf0,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff cf f0 25 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0.d, p7/z, z7.d
// CHECK-INST: movprfx	z0.d, p7/z, z7.d
// CHECK-ENCODING: [0xe0,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 3c d0 04 <unknown>

mul z0.d, p7/m, z0.d, z31.d
// CHECK-INST: mul	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x1f,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 1f d0 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

mul z0.d, p7/m, z0.d, z31.d
// CHECK-INST: mul	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x1f,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 1f d0 04 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: df bc 20 04 <unknown>

mul z31.d, z31.d, #127
// CHECK-INST: mul	z31.d, z31.d, #127
// CHECK-ENCODING: [0xff,0xcf,0xf0,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff cf f0 25 <unknown>
