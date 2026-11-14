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

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A, string Invert = "">
    : CCIf<!strconcat(Invert,
                      "State.getMachineFunction().getSubtarget<SuperHSubtarget>().",
                      F),
           A>;

def CC_SH : CallingConv<[

  // All lower size parameters are promoted to 32-bit
  CCIfType<[i1, i8, i16],  CCPromoteToType<i32>>,
  CCIfType<[i32], CCAssignToReg<[R4, R5, R6, R7]>>,

  // 32-bit floats
  CCIfSubtarget<"hasFPU()", 
      CCIfType<[f32], CCAssignToReg<[FR4,  FR5,  FR6,  FR7,  FR8,  FR9,
                                     FR10, FR11]>>>,

  CCAssignToStack<4, 4>
]>;

def RetCC_SH : CallingConv<[
  CCIfSRet<CCCustom<"RetCC_SH_SRet">>,

  // All lower size parameters are promoted to 32-bit
  CCIfType<[i1, i8, i16],  CCPromoteToType<i32>>,
  CCIfType<[i32],  CCAssignToReg<[R0, R1, R2, R3]>>,
  CCIfType<[i64],  CCAssignToReg<[R0, R1]>>,

  // 32-bit floats
  CCIfSubtarget<"hasFPU()", 
      CCIfType<[f32], CCAssignToReg<[FR0, FR1, FR2, FR3]>>>,

  CCAssignToStack<4, 4>
]>;

//===----------------------------------------------------------------------===//
// Callee-saved register lists.
//===----------------------------------------------------------------------===//

def CSR_NoRegs : CalleeSavedRegs<(add)>;

def CSR_SH : CalleeSavedRegs<(add R8, R9, R10, R11, R12, 
										R13, R14, R15)>;
