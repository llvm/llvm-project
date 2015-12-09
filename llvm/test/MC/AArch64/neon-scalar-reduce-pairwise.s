// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon,+fullfp16 -show-encoding < %s | FileCheck %s

//----------------------------------------------------------------------
// Scalar Reduce Add Pairwise (Integer)
//----------------------------------------------------------------------
      addp d0, v1.2d

// CHECK: addp d0, v1.2d     // encoding: [0x20,0xb8,0xf1,0x5e]

//----------------------------------------------------------------------
// Scalar Reduce Add Pairwise (Floating Point)
//----------------------------------------------------------------------
      faddp h18, v3.2h
      faddp h18, v3.2H
      faddp s19, v2.2s
      faddp d20, v1.2d

// CHECK: faddp h18, v3.2h     // encoding: [0x72,0xd8,0x30,0x5e]
// CHECK: faddp s19, v2.2s     // encoding: [0x53,0xd8,0x30,0x7e]
// CHECK: faddp d20, v1.2d     // encoding: [0x34,0xd8,0x70,0x7e]

