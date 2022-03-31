// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+bf16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve,+bf16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

bfdot z0.S, z1.H, z2.H
// CHECK-INST: bfdot z0.s, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x80,0x62,0x64]
// CHECK-ERROR: instruction requires: bf16 sve or sme

bfdot z0.S, z1.H, z2.H[0]
// CHECK-INST: bfdot z0.s, z1.h, z2.h[0]
// CHECK-ENCODING: [0x20,0x40,0x62,0x64]
// CHECK-ERROR: instruction requires: bf16 sve or sme

bfdot z0.S, z1.H, z2.H[3]
// CHECK-INST: bfdot z0.s, z1.h, z2.h[3]
// CHECK-ENCODING: [0x20,0x40,0x7a,0x64]
// CHECK-ERROR: instruction requires: bf16 sve or sme

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme

bfdot z0.S, z1.H, z2.H
// CHECK-INST: bfdot z0.s, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x80,0x62,0x64]
// CHECK-ERROR: instruction requires: bf16 sve or sme

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme

bfdot z0.S, z1.H, z2.H[0]
// CHECK-INST: bfdot z0.s, z1.h, z2.h[0]
// CHECK-ENCODING: [0x20,0x40,0x62,0x64]
// CHECK-ERROR: instruction requires: bf16 sve or sme

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme

bfdot z0.S, z1.H, z2.H[3]
// CHECK-INST: bfdot z0.s, z1.h, z2.h[3]
// CHECK-ENCODING: [0x20,0x40,0x7a,0x64]
// CHECK-ERROR: instruction requires: bf16 sve or sme
