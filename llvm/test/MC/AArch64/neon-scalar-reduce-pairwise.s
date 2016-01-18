// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

//----------------------------------------------------------------------
// Scalar Reduce Add Pairwise (Integer)
//----------------------------------------------------------------------
      addp d0, v1.2d

// CHECK: addp d0, v1.2d     // encoding: [0x20,0xb8,0xf1,0x5e]

//----------------------------------------------------------------------
// Scalar Reduce Add Pairwise (Floating Point)
//----------------------------------------------------------------------
      faddp d20, v1.2d

// CHECK: faddp d20, v1.2d     // encoding: [0x34,0xd8,0x70,0x7e]

