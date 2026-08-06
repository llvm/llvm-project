//===-- SuperHCallingConv.td - Calling Conventions for SuperH -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//
// This describes the calling conventions for the SuperH architecture.
//===-------------------------------------------------------------------------===//

//===-------------------------------------------------------------------------===//
// SuperH CDECL Calling Convention
//===-------------------------------------------------------------------------===//
let Entry = 1 in
def CC_SH_CDECL : CallingConv<[

  // Handles byval parameters.
  CCIfByVal<CCPassByVal<4, 4>>,
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  CCIfType<[i32], CCAssignToReg<[R4, R5, R6, R7]>>,
  CCIfType<[i32], CCAssignToStack<4, 4>>,
  CCIfType<[f32], CCAssignToStack<4, 4>>,
]>;

def CC_SH_RENESAS : CallingConv<[

  // Handles byval parameters.
  CCIfByVal<CCPassByVal<4, 4>>,
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  CCIfType<[i32], CCAssignToReg<[R4, R5, R6, R7]>>,
  CCIfType<[i32], CCAssignToStack<4, 4>>,
  CCIfType<[f32], CCAssignToStack<4, 4>>,
]>;

def CC_SH_WinCE : CallingConv<[

  // Handles byval parameters.
  CCIfByVal<CCPassByVal<4, 4>>,
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  CCIfType<[i32], CCAssignToReg<[R4, R5, R6, R7]>>,
  CCIfType<[i32], CCAssignToStack<4, 4>>,
  CCIfType<[f32], CCAssignToStack<4, 4>>,
]>;

//===----------------------------------------------------------------------===//
// Callee-saved register lists.
//===----------------------------------------------------------------------===//

def CSR_SH_CDECL : CalleeSavedRegs<(add R8, R9, R10, R11, R12, 
										R13, R14, R15)>;

def CSR_SH_RENESAS : CalleeSavedRegs<(add R4, R5, R6, R7, R8, 
										  R9, R10, R11, R12, R13, 
										  R14, R15)>;

def CSR_SH_WINCE : CalleeSavedRegs<(add R8, R9, R10, R11, R12, 
										R13, R14, R15)>;