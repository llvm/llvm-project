//===- X86CallingConv.td - Calling Conventions X86 32/64 ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the X86-32 and X86-64
// architectures.
//
//===----------------------------------------------------------------------===//

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A>
 : CCIf<!strconcat("State.getTarget().getSubtarget<X86Subtarget>().", F), A>;

//===----------------------------------------------------------------------===//
// Return Value Calling Conventions
//===----------------------------------------------------------------------===//

// Return-value conventions common to all X86 CC's.
def RetCC_X86Common : CallingConv<[
  // Scalar values are returned in AX first, then DX.  For i8, the ABI
  // requires the values to be in AL and AH, however this code uses AL and DL
  // instead. This is because using AH for the second register conflicts with
  // the way LLVM does multiple return values -- a return of {i16,i8} would end
  // up in AX and AH, which overlap. Front-ends wishing to conform to the ABI
  // for functions that return two i8 values are currently expected to pack the
  // values into an i16 (which uses AX, and thus AL:AH).
  CCIfType<[i8] , CCAssignToReg<[AL, DL]>>,
  CCIfType<[i16], CCAssignToReg<[AX, DX]>>,
  CCIfType<[i32], CCAssignToReg<[EAX, EDX]>>,
  CCIfType<[i64], CCAssignToReg<[RAX, RDX]>>,
  
  // Vector types are returned in XMM0 and XMM1, when they fit.  XMMM2 and XMM3
  // can only be used by ABI non-compliant code. If the target doesn't have XMM
  // registers, it won't have vector types.
  CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
            CCAssignToReg<[XMM0,XMM1,XMM2,XMM3]>>,

  // MMX vector types are always returned in MM0. If the target doesn't have
  // MM0, it doesn't support these vector types.
  CCIfType<[v8i8, v4i16, v2i32, v1i64, v2f32], CCAssignToReg<[MM0]>>,

  // Long double types are always returned in ST0 (even with SSE).
  CCIfType<[f80], CCAssignToReg<[ST0, ST1]>>
]>;

// X86-32 C return-value convention.
def RetCC_X86_32_C : CallingConv<[
  // The X86-32 calling convention returns FP values in ST0, unless marked
  // with "inreg" (used here to distinguish one kind of reg from another,
  // weirdly; this is really the sse-regparm calling convention) in which
  // case they use XMM0, otherwise it is the same as the common X86 calling
  // conv.
  CCIfInReg<CCIfSubtarget<"hasSSE2()",
    CCIfType<[f32, f64], CCAssignToReg<[XMM0,XMM1,XMM2]>>>>,
  CCIfType<[f32,f64], CCAssignToReg<[ST0, ST1]>>,
  CCDelegateTo<RetCC_X86Common>
]>;

// X86-32 FastCC return-value convention.
def RetCC_X86_32_Fast : CallingConv<[
  // The X86-32 fastcc returns 1, 2, or 3 FP values in XMM0-2 if the target has
  // SSE2.
  // This can happen when a float, 2 x float, or 3 x float vector is split by
  // target lowering, and is returned in 1-3 sse regs.
  CCIfType<[f32], CCIfSubtarget<"hasSSE2()", CCAssignToReg<[XMM0,XMM1,XMM2]>>>,
  CCIfType<[f64], CCIfSubtarget<"hasSSE2()", CCAssignToReg<[XMM0,XMM1,XMM2]>>>,

  // For integers, ECX can be used as an extra return register
  CCIfType<[i8],  CCAssignToReg<[AL, DL, CL]>>,
  CCIfType<[i16], CCAssignToReg<[AX, DX, CX]>>,
  CCIfType<[i32], CCAssignToReg<[EAX, EDX, ECX]>>,

  // Otherwise, it is the same as the common X86 calling convention.
  CCDelegateTo<RetCC_X86Common>
]>;

// X86-64 C return-value convention.
def RetCC_X86_64_C : CallingConv<[
  // The X86-64 calling convention always returns FP values in XMM0.
  CCIfType<[f32], CCAssignToReg<[XMM0, XMM1]>>,
  CCIfType<[f64], CCAssignToReg<[XMM0, XMM1]>>,

  // MMX vector types are always returned in XMM0 except for v1i64 which is
  // returned in RAX. This disagrees with ABI documentation but is bug
  // compatible with gcc.
  CCIfType<[v1i64], CCAssignToReg<[RAX]>>,
  CCIfType<[v8i8, v4i16, v2i32, v2f32], CCAssignToReg<[XMM0, XMM1]>>,
  CCDelegateTo<RetCC_X86Common>
]>;

// X86-Win64 C return-value convention.
def RetCC_X86_Win64_C : CallingConv<[
  // The X86-Win64 calling convention always returns __m64 values in RAX.
  CCIfType<[v8i8, v4i16, v2i32, v1i64], CCBitConvertToType<i64>>,

  // And FP in XMM0 only.
  CCIfType<[f32], CCAssignToReg<[XMM0]>>,
  CCIfType<[f64], CCAssignToReg<[XMM0]>>,

  // Otherwise, everything is the same as 'normal' X86-64 C CC.
  CCDelegateTo<RetCC_X86_64_C>
]>;


// This is the root return-value convention for the X86-32 backend.
def RetCC_X86_32 : CallingConv<[
  // If FastCC, use RetCC_X86_32_Fast.
  CCIfCC<"CallingConv::Fast", CCDelegateTo<RetCC_X86_32_Fast>>,
  // Otherwise, use RetCC_X86_32_C.
  CCDelegateTo<RetCC_X86_32_C>
]>;

// This is the root return-value convention for the X86-64 backend.
def RetCC_X86_64 : CallingConv<[
  // Mingw64 and native Win64 use Win64 CC
  CCIfSubtarget<"isTargetWin64()", CCDelegateTo<RetCC_X86_Win64_C>>,

  // Otherwise, drop to normal X86-64 CC
  CCDelegateTo<RetCC_X86_64_C>
]>;

// This is the return-value convention used for the entire X86 backend.
def RetCC_X86 : CallingConv<[
  CCIfSubtarget<"is64Bit()", CCDelegateTo<RetCC_X86_64>>,
  CCDelegateTo<RetCC_X86_32>
]>;

//===----------------------------------------------------------------------===//
// X86-64 Argument Calling Conventions
//===----------------------------------------------------------------------===//

def CC_X86_64_C : CallingConv<[
  // Handles byval parameters.
  CCIfByVal<CCPassByVal<8, 8>>,

  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // The 'nest' parameter, if any, is passed in R10.
  CCIfNest<CCAssignToReg<[R10]>>,

  // The first 6 v1i64 vector arguments are passed in GPRs on Darwin.
  CCIfType<[v1i64],
            CCIfSubtarget<"isTargetDarwin()",
            CCBitConvertToType<i64>>>,

  // The first 6 integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToReg<[EDI, ESI, EDX, ECX, R8D, R9D]>>,
  CCIfType<[i64], CCAssignToReg<[RDI, RSI, RDX, RCX, R8 , R9 ]>>,

  // The first 8 MMX (except for v1i64) vector arguments are passed in XMM
  // registers on Darwin.
  CCIfType<[v8i8, v4i16, v2i32, v2f32],
            CCIfSubtarget<"isTargetDarwin()",
            CCIfSubtarget<"hasSSE2()",
            CCPromoteToType<v2i64>>>>,

  // The first 8 FP/Vector arguments are passed in XMM registers.
  CCIfType<[f32, f64, v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
            CCIfSubtarget<"hasSSE1()",
            CCAssignToReg<[XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]>>>,
 
  // Integer/FP values get stored in stack slots that are 8 bytes in size and
  // 8-byte aligned if there are no more registers to hold them.
  CCIfType<[i32, i64, f32, f64], CCAssignToStack<8, 8>>,
  
  // Long doubles get stack slots whose size and alignment depends on the
  // subtarget.
  CCIfType<[f80], CCAssignToStack<0, 0>>,

  // Vectors get 16-byte stack slots that are 16-byte aligned.
  CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64], CCAssignToStack<16, 16>>,

  // __m64 vectors get 8-byte stack slots that are 8-byte aligned.
  CCIfType<[v8i8, v4i16, v2i32, v1i64, v2f32], CCAssignToStack<8, 8>>
]>;

// Calling convention used on Win64
def CC_X86_Win64_C : CallingConv<[
  // FIXME: Handle byval stuff.
  // FIXME: Handle varargs.

  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // The 'nest' parameter, if any, is passed in R10.
  CCIfNest<CCAssignToReg<[R10]>>,

  // 128 bit vectors are passed by pointer
  CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64], CCPassIndirect<i64>>,

  // The first 4 MMX vector arguments are passed in GPRs.
  CCIfType<[v8i8, v4i16, v2i32, v1i64, v2f32],
           CCBitConvertToType<i64>>,

  // The first 4 integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToRegWithShadow<[ECX , EDX , R8D , R9D ],
                                          [XMM0, XMM1, XMM2, XMM3]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[RCX , RDX , R8  , R9  ],
                                          [XMM0, XMM1, XMM2, XMM3]>>,

  // The first 4 FP/Vector arguments are passed in XMM registers.
  CCIfType<[f32, f64, v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
           CCAssignToRegWithShadow<[XMM0, XMM1, XMM2, XMM3],
                                   [RCX , RDX , R8  , R9  ]>>,

  // Integer/FP values get stored in stack slots that are 8 bytes in size and
  // 8-byte aligned if there are no more registers to hold them.
  CCIfType<[i32, i64, f32, f64], CCAssignToStack<8, 8>>,

  // Long doubles get stack slots whose size and alignment depends on the
  // subtarget.
  CCIfType<[f80], CCAssignToStack<0, 0>>,

  // __m64 vectors get 8-byte stack slots that are 8-byte aligned.
  CCIfType<[v8i8, v4i16, v2i32, v1i64], CCAssignToStack<8, 8>>
]>;

def CC_X86_64_GHC : CallingConv<[
  // Promote i8/i16/i32 arguments to i64.
  CCIfType<[i8, i16, i32], CCPromoteToType<i64>>,

  // Pass in STG registers: Base, Sp, Hp, R1, R2, R3, R4, R5, R6, SpLim
  CCIfType<[i64],
            CCAssignToReg<[R13, RBP, R12, RBX, R14, RSI, RDI, R8, R9, R15]>>,

  // Pass in STG registers: F1, F2, F3, F4, D1, D2
  CCIfType<[f32, f64, v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
            CCIfSubtarget<"hasSSE1()",
            CCAssignToReg<[XMM1, XMM2, XMM3, XMM4, XMM5, XMM6]>>>
]>;

//===----------------------------------------------------------------------===//
// X86 C Calling Convention
//===----------------------------------------------------------------------===//

/// CC_X86_32_Common - In all X86-32 calling conventions, extra integers and FP
/// values are spilled on the stack, and the first 4 vector values go in XMM
/// regs.
def CC_X86_32_Common : CallingConv<[
  // Handles byval parameters.
  CCIfByVal<CCPassByVal<4, 4>>,

  // The first 3 float or double arguments, if marked 'inreg' and if the call
  // is not a vararg call and if SSE2 is available, are passed in SSE registers.
  CCIfNotVarArg<CCIfInReg<CCIfType<[f32,f64],
                CCIfSubtarget<"hasSSE2()",
                CCAssignToReg<[XMM0,XMM1,XMM2]>>>>>,

  // The first 3 __m64 (except for v1i64) vector arguments are passed in mmx
  // registers if the call is not a vararg call.
  CCIfNotVarArg<CCIfType<[v8i8, v4i16, v2i32, v2f32],
                CCAssignToReg<[MM0, MM1, MM2]>>>,

  // Integer/Float values get stored in stack slots that are 4 bytes in
  // size and 4-byte aligned.
  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,
  
  // Doubles get 8-byte slots that are 4-byte aligned.
  CCIfType<[f64], CCAssignToStack<8, 4>>,

  // Long doubles get slots whose size depends on the subtarget.
  CCIfType<[f80], CCAssignToStack<0, 4>>,

  // The first 4 SSE vector arguments are passed in XMM registers.
  CCIfNotVarArg<CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
                CCAssignToReg<[XMM0, XMM1, XMM2, XMM3]>>>,

  // Other SSE vectors get 16-byte stack slots that are 16-byte aligned.
  CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64], CCAssignToStack<16, 16>>,

  // __m64 vectors get 8-byte stack slots that are 4-byte aligned. They are
  // passed in the parameter area.
  CCIfType<[v8i8, v4i16, v2i32, v1i64], CCAssignToStack<8, 4>>]>;

def CC_X86_32_C : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // The 'nest' parameter, if any, is passed in ECX.
  CCIfNest<CCAssignToReg<[ECX]>>,

  // The first 3 integer arguments, if marked 'inreg' and if the call is not
  // a vararg call, are passed in integer registers.
  CCIfNotVarArg<CCIfInReg<CCIfType<[i32], CCAssignToReg<[EAX, EDX, ECX]>>>>,

  // Otherwise, same as everything else.
  CCDelegateTo<CC_X86_32_Common>
]>;

def CC_X86_32_FastCall : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // The 'nest' parameter, if any, is passed in EAX.
  CCIfNest<CCAssignToReg<[EAX]>>,

  // The first 2 integer arguments are passed in ECX/EDX
  CCIfType<[i32], CCAssignToReg<[ECX, EDX]>>,

  // Otherwise, same as everything else.
  CCDelegateTo<CC_X86_32_Common>
]>;

def CC_X86_32_FastCC : CallingConv<[
  // Handles byval parameters.  Note that we can't rely on the delegation
  // to CC_X86_32_Common for this because that happens after code that
  // puts arguments in registers.
  CCIfByVal<CCPassByVal<4, 4>>,

  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // The 'nest' parameter, if any, is passed in EAX.
  CCIfNest<CCAssignToReg<[EAX]>>,

  // The first 2 integer arguments are passed in ECX/EDX
  CCIfType<[i32], CCAssignToReg<[ECX, EDX]>>,

  // The first 3 float or double arguments, if the call is not a vararg
  // call and if SSE2 is available, are passed in SSE registers.
  CCIfNotVarArg<CCIfType<[f32,f64],
                CCIfSubtarget<"hasSSE2()",
                CCAssignToReg<[XMM0,XMM1,XMM2]>>>>,

  // Doubles get 8-byte slots that are 8-byte aligned.
  CCIfType<[f64], CCAssignToStack<8, 8>>,

  // Otherwise, same as everything else.
  CCDelegateTo<CC_X86_32_Common>
]>;

def CC_X86_32_GHC : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // Pass in STG registers: Base, Sp, Hp, R1
  CCIfType<[i32], CCAssignToReg<[EBX, EBP, EDI, ESI]>>
]>;
