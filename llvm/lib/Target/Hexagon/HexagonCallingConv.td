//===- HexagonCallingConv.td - Calling Conventions Hexagon -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the Hexagon architectures.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Return Value Calling Conventions
//===----------------------------------------------------------------------===//

// Hexagon 32-bit C return-value convention.
def RetCC_Hexagon32 : CallingConv<[
  CCIfType<[i32, f32], CCAssignToReg<[R0, R1, R2, R3, R4, R5]>>,
  CCIfType<[i64, f64], CCAssignToReg<[D0, D1, D2]>>,

  // Alternatively, they are assigned to the stack in 4-byte aligned units.
  CCAssignToStack<4, 4>
]>;

// Hexagon 32-bit C Calling convention.
def CC_Hexagon32 : CallingConv<[
  // All arguments get passed in integer registers if there is space.
  CCIfType<[f32, i32, i16, i8], CCAssignToReg<[R0, R1, R2, R3, R4, R5]>>,
  CCIfType<[f64, i64], CCAssignToReg<[D0, D1, D2]>>,

  // Alternatively, they are assigned to the stack in 4-byte aligned units.
  CCAssignToStack<4, 4>
]>;
