//===- MipsCallingConv.td - Calling Conventions for Mips --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// This describes the calling conventions for Mips architecture.
//===----------------------------------------------------------------------===//

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A>: 
  CCIf<!strconcat("State.getTarget().getSubtarget<MipsSubtarget>().", F), A>;

//===----------------------------------------------------------------------===//
// Mips O32 Calling Convention
//===----------------------------------------------------------------------===//
def CC_MipsO32 : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // The first 4 integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToReg<[A0, A1, A2, A3]>>,

  // Integer values get stored in stack slots that are 4 bytes in
  // size and 4-byte aligned.
  CCIfType<[i32], CCAssignToStack<4, 4>>
]>;

def RetCC_MipsO32 : CallingConv<[
  // i32 are returned in registers V0, V1
  CCIfType<[i32], CCAssignToReg<[V0, V1]>>,

  // f32 are returned in registers F0, F1
  CCIfType<[f32], CCAssignToReg<[F0, F1]>>,

  // f64 are returned in register D0
  CCIfType<[f64], CCIfSubtarget<"isNotSingleFloat()", CCAssignToReg<[D0]>>>
]>;

//===----------------------------------------------------------------------===//
// Mips EABI Calling Convention
//===----------------------------------------------------------------------===//
def CC_MipsEABI : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // Integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToReg<[A0, A1, A2, A3, T0, T1, T2, T3]>>,

  // Single fp arguments are passed in pairs within 32-bit mode 
  CCIfType<[f32], CCIfSubtarget<"isSingleFloat()", 
                  CCAssignToReg<[F12, F13, F14, F15, F16, F17, F18, F19]>>>,

  CCIfType<[f32], CCIfSubtarget<"isNotSingleFloat()", 
                  CCAssignToReg<[F12, F14, F16, F18]>>>,

  // The first 4 doubl fp arguments are passed in single fp registers.
  CCIfType<[f64], CCIfSubtarget<"isNotSingleFloat()", 
                  CCAssignToReg<[D6, D7, D8, D9]>>>,

  // Integer values get stored in stack slots that are 4 bytes in
  // size and 4-byte aligned.
  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,

  // Integer values get stored in stack slots that are 8 bytes in
  // size and 8-byte aligned.
  CCIfType<[f64], CCIfSubtarget<"isNotSingleFloat()", CCAssignToStack<8, 8>>>
]>;

def RetCC_MipsEABI : CallingConv<[
  // i32 are returned in registers V0, V1
  CCIfType<[i32], CCAssignToReg<[V0, V1]>>,

  // f32 are returned in registers F0, F1
  CCIfType<[f32], CCAssignToReg<[F0, F1]>>,

  // f64 are returned in register D0
  CCIfType<[f64], CCIfSubtarget<"isNotSingleFloat()", CCAssignToReg<[D0]>>>
]>;

//===----------------------------------------------------------------------===//
// Mips Calling Convention Dispatch
//===----------------------------------------------------------------------===//

def CC_Mips : CallingConv<[
  CCIfSubtarget<"isABI_EABI()", CCDelegateTo<CC_MipsEABI>>,
  CCDelegateTo<CC_MipsO32>
]>;

def RetCC_Mips : CallingConv<[
  CCIfSubtarget<"isABI_EABI()", CCDelegateTo<RetCC_MipsEABI>>,
  CCDelegateTo<RetCC_MipsO32>
]>;
