// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+bf16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

bfmmla z0.S, z1.H, z2.H
// CHECK-INST: bfmmla z0.s, z1.h, z2.h
// CHECK-ENCODING: [0x20,0xe4,0x62,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmmla z0.S, z1.H, z2.H
// CHECK-INST: bfmmla z0.s, z1.h, z2.h
// CHECK-ENCODING: [0x20,0xe4,0x62,0x64]
// CHECK-ERROR: instruction requires: bf16 sve
