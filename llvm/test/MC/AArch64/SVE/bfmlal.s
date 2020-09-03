// RUN: llvm-mc -o - -triple=aarch64 -show-encoding -mattr=+sve,+bf16 %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -o - -triple=aarch64 -show-encoding %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

bfmlalb z0.S, z1.H, z2.H
// CHECK-INST: bfmlalb z0.s, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x80,0xe2,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

bfmlalt z0.S, z1.H, z2.H
// CHECK-INST: bfmlalt z0.s, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x84,0xe2,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

bfmlalb z0.S, z1.H, z2.H[0]
// CHECK-INST: bfmlalb z0.s, z1.h, z2.h[0]
// CHECK-ENCODING: [0x20,0x40,0xe2,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

bfmlalt z0.S, z1.H, z2.H[0]
// CHECK-INST: bfmlalt z0.s, z1.h, z2.h[0]
// CHECK-ENCODING: [0x20,0x44,0xe2,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

bfmlalb z0.S, z1.H, z2.H[7]
// CHECK-INST: bfmlalb z0.s, z1.h, z2.h[7]
// CHECK-ENCODING: [0x20,0x48,0xfa,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

bfmlalt z0.S, z1.H, z2.H[7]
// CHECK-INST: bfmlalt z0.s, z1.h, z2.h[7]
// CHECK-ENCODING: [0x20,0x4c,0xfa,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

bfmlalt z0.S, z1.H, z7.H[7]
// CHECK-INST: bfmlalt z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0x4c,0xff,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

bfmlalb z10.S, z21.H, z14.H
// CHECK-INST: bfmlalb z10.s, z21.h, z14.h
// CHECK-ENCODING: [0xaa,0x82,0xee,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

bfmlalt z14.S, z10.H, z21.H
// CHECK-INST: bfmlalt z14.s, z10.h, z21.h
// CHECK-ENCODING: [0x4e,0x85,0xf5,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

bfmlalb z21.s, z14.h, z3.h[2]
// CHECK-INST: bfmlalb z21.s, z14.h, z3.h[2]
// CHECK-ENCODING: [0xd5,0x41,0xeb,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmlalb z0.S, z1.H, z2.H
// CHECK-INST: bfmlalb z0.s, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x80,0xe2,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmlalt z0.S, z1.H, z2.H
// CHECK-INST: bfmlalt z0.s, z1.h, z2.h
// CHECK-ENCODING: [0x20,0x84,0xe2,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmlalb z0.S, z1.H, z2.H[0]
// CHECK-INST: bfmlalb z0.s, z1.h, z2.h[0]
// CHECK-ENCODING: [0x20,0x40,0xe2,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmlalt z0.S, z1.H, z2.H[0]
// CHECK-INST: bfmlalt z0.s, z1.h, z2.h[0]
// CHECK-ENCODING: [0x20,0x44,0xe2,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmlalb z0.S, z1.H, z2.H[7]
// CHECK-INST: bfmlalb z0.s, z1.h, z2.h[7]
// CHECK-ENCODING: [0x20,0x48,0xfa,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmlalt z0.S, z1.H, z2.H[7]
// CHECK-INST: bfmlalt z0.s, z1.h, z2.h[7]
// CHECK-ENCODING: [0x20,0x4c,0xfa,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

movprfx z0, z7
// CHECK-INST: movprfx z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmlalt z0.S, z1.H, z7.H[7]
// CHECK-INST: bfmlalt z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0x4c,0xff,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

movprfx z10, z7
// CHECK-INST: movprfx z10, z7
// CHECK-ENCODING: [0xea,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmlalb z10.S, z21.H, z14.H
// CHECK-INST: bfmlalb z10.s, z21.h, z14.h
// CHECK-ENCODING: [0xaa,0x82,0xee,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

movprfx z14, z7
// CHECK-INST: movprfx z14, z7
// CHECK-ENCODING: [0xee,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmlalt z14.S, z10.H, z21.H
// CHECK-INST: bfmlalt z14.s, z10.h, z21.h
// CHECK-ENCODING: [0x4e,0x85,0xf5,0x64]
// CHECK-ERROR: instruction requires: bf16 sve

movprfx z21, z7
// CHECK-INST: movprfx z21, z7
// CHECK-ENCODING: [0xf5,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve

bfmlalb z21.s, z14.h, z3.h[2]
// CHECK-INST: bfmlalb z21.s, z14.h, z3.h[2]
// CHECK-ENCODING: [0xd5,0x41,0xeb,0x64]
// CHECK-ERROR: instruction requires: bf16 sve
