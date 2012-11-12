//===-- MipsCallingConv.td - Calling Conventions for Mips --*- tablegen -*-===//
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

// Only the return rules are defined here for O32. The rules for argument
// passing are defined in MipsISelLowering.cpp.
def RetCC_MipsO32 : CallingConv<[
  // i32 are returned in registers V0, V1, A0, A1
  CCIfType<[i32], CCAssignToReg<[V0, V1, A0, A1]>>,

  // f32 are returned in registers F0, F2
  CCIfType<[f32], CCAssignToReg<[F0, F2]>>,

  // f64 are returned in register D0, D1
  CCIfType<[f64], CCIfSubtarget<"isNotSingleFloat()", CCAssignToReg<[D0, D1]>>>
]>;

//===----------------------------------------------------------------------===//
// Mips N32/64 Calling Convention
//===----------------------------------------------------------------------===//

def CC_MipsN : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // Integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToRegWithShadow<[A0, A1, A2, A3,
                                           T0, T1, T2, T3],
                                          [F12, F13, F14, F15,
                                           F16, F17, F18, F19]>>,

  CCIfType<[i64], CCAssignToRegWithShadow<[A0_64, A1_64, A2_64, A3_64,
                                           T0_64, T1_64, T2_64, T3_64],
                                          [D12_64, D13_64, D14_64, D15_64,
                                           D16_64, D17_64, D18_64, D19_64]>>,

  // f32 arguments are passed in single precision FP registers.
  CCIfType<[f32], CCAssignToRegWithShadow<[F12, F13, F14, F15,
                                           F16, F17, F18, F19],
                                          [A0_64, A1_64, A2_64, A3_64,
                                           T0_64, T1_64, T2_64, T3_64]>>,

  // f64 arguments are passed in double precision FP registers.
  CCIfType<[f64], CCAssignToRegWithShadow<[D12_64, D13_64, D14_64, D15_64,
                                           D16_64, D17_64, D18_64, D19_64],
                                          [A0_64, A1_64, A2_64, A3_64,
                                           T0_64, T1_64, T2_64, T3_64]>>,

  // All stack parameter slots become 64-bit doublewords and are 8-byte aligned.
  CCIfType<[i32, f32], CCAssignToStack<4, 8>>,
  CCIfType<[i64, f64], CCAssignToStack<8, 8>>
]>;

// N32/64 variable arguments.
// All arguments are passed in integer registers.
def CC_MipsN_VarArg : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  CCIfType<[i32, f32], CCAssignToReg<[A0, A1, A2, A3, T0, T1, T2, T3]>>,

  CCIfType<[i64, f64], CCAssignToReg<[A0_64, A1_64, A2_64, A3_64,
                                      T0_64, T1_64, T2_64, T3_64]>>,

  // All stack parameter slots become 64-bit doublewords and are 8-byte aligned.
  CCIfType<[i32, f32], CCAssignToStack<4, 8>>,
  CCIfType<[i64, f64], CCAssignToStack<8, 8>>
]>;

def RetCC_MipsN : CallingConv<[
  // i32 are returned in registers V0, V1
  CCIfType<[i32], CCAssignToReg<[V0, V1]>>,

  // i64 are returned in registers V0_64, V1_64
  CCIfType<[i64], CCAssignToReg<[V0_64, V1_64]>>,

  // f32 are returned in registers F0, F2
  CCIfType<[f32], CCAssignToReg<[F0, F2]>>,

  // f64 are returned in registers D0, D2
  CCIfType<[f64], CCAssignToReg<[D0_64, D2_64]>>
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

  // The first 4 double fp arguments are passed in single fp registers.
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
// Mips Android Calling Convention
//===----------------------------------------------------------------------===//

def RetCC_MipsAndroid : CallingConv<[
  // f32 are returned in registers F0, F2, F1, F3
  CCIfType<[f32], CCAssignToReg<[F0, F2, F1, F3]>>,

  CCDelegateTo<RetCC_MipsO32>
]>;

//===----------------------------------------------------------------------===//
// Mips FastCC Calling Convention
//===----------------------------------------------------------------------===//
def CC_MipsO32_FastCC : CallingConv<[
  // f64 arguments are passed in double-precision floating pointer registers.
  CCIfType<[f64], CCAssignToReg<[D0, D1, D2, D3, D4, D5, D6, D7, D8, D9]>>,

  // Stack parameter slots for f64 are 64-bit doublewords and 8-byte aligned.
  CCIfType<[f64], CCAssignToStack<8, 8>>
]>;

def CC_MipsN_FastCC : CallingConv<[
  // Integer arguments are passed in integer registers.
  CCIfType<[i64], CCAssignToReg<[A0_64, A1_64, A2_64, A3_64, T0_64, T1_64,
                                 T2_64, T3_64, T4_64, T5_64, T6_64, T7_64,
                                 T8_64, V1_64]>>,

  // f64 arguments are passed in double-precision floating pointer registers.
  CCIfType<[f64], CCAssignToReg<[D0_64, D1_64, D2_64, D3_64, D4_64, D5_64,
                                 D6_64, D7_64, D8_64, D9_64, D10_64, D11_64,
                                 D12_64, D13_64, D14_64, D15_64, D16_64, D17_64,
                                 D18_64, D19_64]>>,

  // Stack parameter slots for i64 and f64 are 64-bit doublewords and
  // 8-byte aligned.
  CCIfType<[i64, f64], CCAssignToStack<8, 8>>
]>;

def CC_Mips_FastCC : CallingConv<[
  // Handles byval parameters.
  CCIfByVal<CCPassByVal<4, 4>>,

  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // Integer arguments are passed in integer registers. All scratch registers,
  // except for AT, V0 and T9, are available to be used as argument registers.
  CCIfType<[i32], CCAssignToReg<[A0, A1, A2, A3, T0, T1, T2, T3, T4, T5, T6,
                                 T7, T8, V1]>>,

  // f32 arguments are passed in single-precision floating pointer registers.
  CCIfType<[f32], CCAssignToReg<[F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                                 F11, F12, F13, F14, F15, F16, F17, F18, F19]>>,

  // Stack parameter slots for i32 and f32 are 32-bit words and 4-byte aligned.
  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,

  CCIfSubtarget<"isABI_EABI()", CCDelegateTo<CC_MipsEABI>>,
  CCIfSubtarget<"isABI_O32()", CCDelegateTo<CC_MipsO32_FastCC>>,
  CCDelegateTo<CC_MipsN_FastCC>
]>;

//===----------------------------------------------------------------------===//
// Mips Calling Convention Dispatch
//===----------------------------------------------------------------------===//

def RetCC_Mips : CallingConv<[
  CCIfSubtarget<"isABI_EABI()", CCDelegateTo<RetCC_MipsEABI>>,
  CCIfSubtarget<"isABI_N32()", CCDelegateTo<RetCC_MipsN>>,
  CCIfSubtarget<"isABI_N64()", CCDelegateTo<RetCC_MipsN>>,
  CCIfSubtarget<"isAndroid()", CCDelegateTo<RetCC_MipsAndroid>>,
  CCDelegateTo<RetCC_MipsO32>
]>;

//===----------------------------------------------------------------------===//
// Callee-saved register lists.
//===----------------------------------------------------------------------===//

def CSR_SingleFloatOnly : CalleeSavedRegs<(add (sequence "F%u", 31, 20), RA, FP,
                                               (sequence "S%u", 7, 0))>;

def CSR_O32 : CalleeSavedRegs<(add (sequence "D%u", 15, 10), RA, FP,
                                   (sequence "S%u", 7, 0))>;

def CSR_N32 : CalleeSavedRegs<(add D31_64, D29_64, D27_64, D25_64, D24_64,
                                   D23_64, D22_64, D21_64, RA_64, FP_64, GP_64,
                                   (sequence "S%u_64", 7, 0))>;

def CSR_N64 : CalleeSavedRegs<(add (sequence "D%u_64", 31, 24), RA_64, FP_64,
                                   GP_64, (sequence "S%u_64", 7, 0))>;
