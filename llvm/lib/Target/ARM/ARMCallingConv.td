//===- ARMCallingConv.td - Calling Conventions for ARM -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for ARM architecture.
//===----------------------------------------------------------------------===//

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A>:
  CCIf<!strconcat("State.getTarget().getSubtarget<ARMSubtarget>().", F), A>;

/// CCIfAlign - Match of the original alignment of the arg
class CCIfAlign<string Align, CCAction A>:
  CCIf<!strconcat("ArgFlags.getOrigAlign() == ", Align), A>;

//===----------------------------------------------------------------------===//
// ARM APCS Calling Convention
//===----------------------------------------------------------------------===//
def CC_ARM_APCS : CallingConv<[

  // Handles byval parameters.
  CCIfByVal<CCPassByVal<4, 4>>,
    
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // Handle all vector types as either f64 or v2f64.
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v2f32], CCBitConvertToType<f64>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32], CCBitConvertToType<v2f64>>,

  // f64 and v2f64 are passed in adjacent GPRs, possibly split onto the stack
  CCIfType<[f64, v2f64], CCCustom<"CC_ARM_APCS_Custom_f64">>,

  CCIfType<[f32], CCBitConvertToType<i32>>,
  CCIfType<[i32], CCAssignToReg<[R0, R1, R2, R3]>>,

  CCIfType<[i32], CCAssignToStack<4, 4>>,
  CCIfType<[f64], CCAssignToStack<8, 4>>,
  CCIfType<[v2f64], CCAssignToStack<16, 4>>
]>;

def RetCC_ARM_APCS : CallingConv<[
  CCIfType<[f32], CCBitConvertToType<i32>>,

  // Handle all vector types as either f64 or v2f64.
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v2f32], CCBitConvertToType<f64>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32], CCBitConvertToType<v2f64>>,

  CCIfType<[f64, v2f64], CCCustom<"RetCC_ARM_APCS_Custom_f64">>,

  CCIfType<[i32], CCAssignToReg<[R0, R1, R2, R3]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[R0, R2], [R1, R3]>>
]>;

//===----------------------------------------------------------------------===//
// ARM APCS Calling Convention for FastCC (when VFP2 or later is available)
//===----------------------------------------------------------------------===//
def FastCC_ARM_APCS : CallingConv<[
  // Handle all vector types as either f64 or v2f64.
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v2f32], CCBitConvertToType<f64>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32], CCBitConvertToType<v2f64>>,

  CCIfType<[v2f64], CCAssignToReg<[Q0, Q1, Q2, Q3]>>,
  CCIfType<[f64], CCAssignToReg<[D0, D1, D2, D3, D4, D5, D6, D7]>>,
  CCIfType<[f32], CCAssignToReg<[S0, S1, S2, S3, S4, S5, S6, S7, S8,
                                 S9, S10, S11, S12, S13, S14, S15]>>,
  CCDelegateTo<CC_ARM_APCS>
]>;

def RetFastCC_ARM_APCS : CallingConv<[
  // Handle all vector types as either f64 or v2f64.
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v2f32], CCBitConvertToType<f64>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32], CCBitConvertToType<v2f64>>,

  CCIfType<[v2f64], CCAssignToReg<[Q0, Q1, Q2, Q3]>>,
  CCIfType<[f64], CCAssignToReg<[D0, D1, D2, D3, D4, D5, D6, D7]>>,
  CCIfType<[f32], CCAssignToReg<[S0, S1, S2, S3, S4, S5, S6, S7, S8,
                                 S9, S10, S11, S12, S13, S14, S15]>>,
  CCDelegateTo<RetCC_ARM_APCS>
]>;

//===----------------------------------------------------------------------===//
// ARM APCS Calling Convention for GHC
//===----------------------------------------------------------------------===//

def CC_ARM_APCS_GHC : CallingConv<[
  // Handle all vector types as either f64 or v2f64.
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v2f32], CCBitConvertToType<f64>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32], CCBitConvertToType<v2f64>>,

  CCIfType<[v2f64], CCAssignToReg<[Q4, Q5]>>,
  CCIfType<[f64], CCAssignToReg<[D8, D9, D10, D11]>>,
  CCIfType<[f32], CCAssignToReg<[S16, S17, S18, S19, S20, S21, S22, S23]>>,

  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // Pass in STG registers: Base, Sp, Hp, R1, R2, R3, R4, SpLim
  CCIfType<[i32], CCAssignToReg<[R4, R5, R6, R7, R8, R9, R10, R11]>>
]>;

//===----------------------------------------------------------------------===//
// ARM AAPCS (EABI) Calling Convention, common parts
//===----------------------------------------------------------------------===//

def CC_ARM_AAPCS_Common : CallingConv<[

  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // i64/f64 is passed in even pairs of GPRs
  // i64 is 8-aligned i32 here, so we may need to eat R1 as a pad register
  // (and the same is true for f64 if VFP is not enabled)
  CCIfType<[i32], CCIfAlign<"8", CCAssignToRegWithShadow<[R0, R2], [R0, R1]>>>,
  CCIfType<[i32], CCIf<"State.getNextStackOffset() == 0 &&"
                       "ArgFlags.getOrigAlign() != 8",
                       CCAssignToReg<[R0, R1, R2, R3]>>>,

  CCIfType<[i32], CCIfAlign<"8", CCAssignToStackWithShadow<4, 8, R3>>>,
  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,
  CCIfType<[f64], CCAssignToStack<8, 8>>,
  CCIfType<[v2f64], CCAssignToStack<16, 8>>
]>;

def RetCC_ARM_AAPCS_Common : CallingConv<[
  CCIfType<[i32], CCAssignToReg<[R0, R1, R2, R3]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[R0, R2], [R1, R3]>>
]>;

//===----------------------------------------------------------------------===//
// ARM AAPCS (EABI) Calling Convention
//===----------------------------------------------------------------------===//

def CC_ARM_AAPCS : CallingConv<[
  // Handle all vector types as either f64 or v2f64.
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v2f32], CCBitConvertToType<f64>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32], CCBitConvertToType<v2f64>>,

  CCIfType<[f64, v2f64], CCCustom<"CC_ARM_AAPCS_Custom_f64">>,
  CCIfType<[f32], CCBitConvertToType<i32>>,
  CCDelegateTo<CC_ARM_AAPCS_Common>
]>;

def RetCC_ARM_AAPCS : CallingConv<[
  // Handle all vector types as either f64 or v2f64.
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v2f32], CCBitConvertToType<f64>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32], CCBitConvertToType<v2f64>>,

  CCIfType<[f64, v2f64], CCCustom<"RetCC_ARM_AAPCS_Custom_f64">>,
  CCIfType<[f32], CCBitConvertToType<i32>>,
  CCDelegateTo<RetCC_ARM_AAPCS_Common>
]>;

//===----------------------------------------------------------------------===//
// ARM AAPCS-VFP (EABI) Calling Convention
// Also used for FastCC (when VFP2 or later is available)
//===----------------------------------------------------------------------===//

def CC_ARM_AAPCS_VFP : CallingConv<[
  // Handle all vector types as either f64 or v2f64.
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v2f32], CCBitConvertToType<f64>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32], CCBitConvertToType<v2f64>>,

  CCIfType<[v2f64], CCAssignToReg<[Q0, Q1, Q2, Q3]>>,
  CCIfType<[f64], CCAssignToReg<[D0, D1, D2, D3, D4, D5, D6, D7]>>,
  CCIfType<[f32], CCAssignToReg<[S0, S1, S2, S3, S4, S5, S6, S7, S8,
                                 S9, S10, S11, S12, S13, S14, S15]>>,
  CCDelegateTo<CC_ARM_AAPCS_Common>
]>;

def RetCC_ARM_AAPCS_VFP : CallingConv<[
  // Handle all vector types as either f64 or v2f64.
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v2f32], CCBitConvertToType<f64>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32], CCBitConvertToType<v2f64>>,

  CCIfType<[v2f64], CCAssignToReg<[Q0, Q1, Q2, Q3]>>,
  CCIfType<[f64], CCAssignToReg<[D0, D1, D2, D3, D4, D5, D6, D7]>>,
  CCIfType<[f32], CCAssignToReg<[S0, S1, S2, S3, S4, S5, S6, S7, S8,
                                 S9, S10, S11, S12, S13, S14, S15]>>,
  CCDelegateTo<RetCC_ARM_AAPCS_Common>
]>;
