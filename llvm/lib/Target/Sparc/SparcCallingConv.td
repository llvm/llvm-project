//===- SparcCallingConv.td - Calling Conventions Sparc -----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the Sparc architectures.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Return Value Calling Conventions
//===----------------------------------------------------------------------===//

// Sparc 32-bit C return-value convention.
def RetCC_Sparc32 : CallingConv<[
  CCIfType<[i32], CCAssignToReg<[I0, I1, I2, I3, I4, I5]>>,
  CCIfType<[f32], CCAssignToReg<[F0, F1, F2, F3]>>,
  CCIfType<[f64], CCAssignToReg<[D0, D1]>>
]>;

// Sparc 32-bit C Calling convention.
def CC_Sparc32 : CallingConv<[
  // All arguments get passed in integer registers if there is space.
  CCIfType<[i32, f32, f64], CCAssignToReg<[I0, I1, I2, I3, I4, I5]>>,
  
  // Alternatively, they are assigned to the stack in 4-byte aligned units.
  CCAssignToStack<4, 4>
]>;
