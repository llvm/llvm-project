//===- Nios2CallingConv.td - Calling Conventions for Nios2 -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for Nios2 architecture.
//===----------------------------------------------------------------------===//

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A>:
  CCIf<!strconcat("State.getTarget().getSubtarget<Nios2Subtarget>().", F), A>;

def CC_Nios2 : CallingConv<[
  // i32 f32 arguments get passed in integer registers if there is space.
  CCIfType<[i32, f32], CCAssignToReg<[R4, R5, R6, R7]>>,

  // Alternatively, they are assigned to the stack in 4-byte aligned units.
  CCAssignToStack<4, 4>
]>;

def RetCC_Nios2EABI : CallingConv<[
  // i32 are returned in registers R2, R3
  CCIfType<[i32], CCAssignToReg<[R2, R3]>>,
  // In case of floating point (FPH2 instr.) also use the same register set
  CCIfType<[f32], CCAssignToReg<[R2, R3]>>,
  CCIfByVal<CCPassByVal<4, 4>>,
  // Stack parameter slots for i32 is 32-bit words and 4-byte aligned.
  CCIfType<[i32], CCAssignToStack<4, 4>>
]>;

def CSR : CalleeSavedRegs<(add RA, FP, (sequence "R%u", 16, 23))>;
