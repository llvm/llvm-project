//===-- X86CallingConv.td - Calling Conventions X86 32/64 --*- tablegen -*-===//
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
  //
  // For code that doesn't care about the ABI, we allow returning more than two
  // integer values in registers.
  CCIfType<[i8] , CCAssignToReg<[AL, DL, CL]>>,
  CCIfType<[i16], CCAssignToReg<[AX, DX, CX]>>,
  CCIfType<[i32], CCAssignToReg<[EAX, EDX, ECX]>>,
  CCIfType<[i64], CCAssignToReg<[RAX, RDX, RCX]>>,

  // Vector types are returned in XMM0 and XMM1, when they fit.  XMM2 and XMM3
  // can only be used by ABI non-compliant code. If the target doesn't have XMM
  // registers, it won't have vector types.
  CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
            CCAssignToReg<[XMM0,XMM1,XMM2,XMM3]>>,

  // 256-bit vectors are returned in YMM0 and XMM1, when they fit. YMM2 and YMM3
  // can only be used by ABI non-compliant code. This vector type is only
  // supported while using the AVX target feature.
  CCIfType<[v32i8, v16i16, v8i32, v4i64, v8f32, v4f64],
            CCAssignToReg<[YMM0,YMM1,YMM2,YMM3]>>,

  // MMX vector types are always returned in MM0. If the target doesn't have
  // MM0, it doesn't support these vector types.
  CCIfType<[x86mmx], CCAssignToReg<[MM0]>>,

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

// Intel_OCL_BI return-value convention.
def RetCC_Intel_OCL_BI : CallingConv<[
  // Vector types are returned in XMM0,XMM1,XMMM2 and XMM3.
  CCIfType<[f32, f64, v4i32, v2i64, v4f32, v2f64],
            CCAssignToReg<[XMM0,XMM1,XMM2,XMM3]>>,

  // 256-bit FP vectors
  // No more than 4 registers
  CCIfType<[v8f32, v4f64, v8i32, v4i64],
            CCAssignToReg<[YMM0,YMM1,YMM2,YMM3]>>,

  // i32, i64 in the standard way
  CCDelegateTo<RetCC_X86Common>
]>;

// X86-64 C return-value convention.
def RetCC_X86_64_C : CallingConv<[
  // The X86-64 calling convention always returns FP values in XMM0.
  CCIfType<[f32], CCAssignToReg<[XMM0, XMM1]>>,
  CCIfType<[f64], CCAssignToReg<[XMM0, XMM1]>>,

  // MMX vector types are always returned in XMM0.
  CCIfType<[x86mmx], CCAssignToReg<[XMM0, XMM1]>>,
  CCDelegateTo<RetCC_X86Common>
]>;

// X86-Win64 C return-value convention.
def RetCC_X86_Win64_C : CallingConv<[
  // The X86-Win64 calling convention always returns __m64 values in RAX.
  CCIfType<[x86mmx], CCBitConvertToType<i64>>,

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

  // Check if this is the Intel OpenCL built-ins calling convention
  CCIfCC<"CallingConv::Intel_OCL_BI", CCDelegateTo<RetCC_Intel_OCL_BI>>,

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

  // The first 6 integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToReg<[EDI, ESI, EDX, ECX, R8D, R9D]>>,
  CCIfType<[i64], CCAssignToReg<[RDI, RSI, RDX, RCX, R8 , R9 ]>>,

  // The first 8 MMX vector arguments are passed in XMM registers on Darwin.
  CCIfType<[x86mmx],
            CCIfSubtarget<"isTargetDarwin()",
            CCIfSubtarget<"hasSSE2()",
            CCPromoteToType<v2i64>>>>,

  // The first 8 FP/Vector arguments are passed in XMM registers.
  CCIfType<[f32, f64, v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
            CCIfSubtarget<"hasSSE1()",
            CCAssignToReg<[XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]>>>,

  // The first 8 256-bit vector arguments are passed in YMM registers, unless
  // this is a vararg function.
  // FIXME: This isn't precisely correct; the x86-64 ABI document says that
  // fixed arguments to vararg functions are supposed to be passed in
  // registers.  Actually modeling that would be a lot of work, though.
  CCIfNotVarArg<CCIfType<[v32i8, v16i16, v8i32, v4i64, v8f32, v4f64],
                          CCIfSubtarget<"hasAVX()",
                          CCAssignToReg<[YMM0, YMM1, YMM2, YMM3,
                                         YMM4, YMM5, YMM6, YMM7]>>>>,

  // Integer/FP values get stored in stack slots that are 8 bytes in size and
  // 8-byte aligned if there are no more registers to hold them.
  CCIfType<[i32, i64, f32, f64], CCAssignToStack<8, 8>>,

  // Long doubles get stack slots whose size and alignment depends on the
  // subtarget.
  CCIfType<[f80], CCAssignToStack<0, 0>>,

  // Vectors get 16-byte stack slots that are 16-byte aligned.
  CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64], CCAssignToStack<16, 16>>,

  // 256-bit vectors get 32-byte stack slots that are 32-byte aligned.
  CCIfType<[v32i8, v16i16, v8i32, v4i64, v8f32, v4f64],
           CCAssignToStack<32, 32>>
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


  // 256 bit vectors are passed by pointer
  CCIfType<[v32i8, v16i16, v8i32, v4i64, v8f32, v4f64], CCPassIndirect<i64>>,

  // The first 4 MMX vector arguments are passed in GPRs.
  CCIfType<[x86mmx], CCBitConvertToType<i64>>,

  // The first 4 integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToRegWithShadow<[ECX , EDX , R8D , R9D ],
                                          [XMM0, XMM1, XMM2, XMM3]>>,
  
  // Do not pass the sret argument in RCX, the Win64 thiscall calling
  // convention requires "this" to be passed in RCX.                                        
  CCIfCC<"CallingConv::X86_ThisCall", 
    CCIfSRet<CCIfType<[i64], CCAssignToRegWithShadow<[RDX , R8  , R9  ],
                                                     [XMM1, XMM2, XMM3]>>>>,

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
  CCIfType<[f80], CCAssignToStack<0, 0>>
]>;

// X86-64 Intel OpenCL built-ins calling convention.
def CC_Intel_OCL_BI : CallingConv<[
  CCIfType<[i32], CCIfSubtarget<"isTargetWin32()", CCAssignToStack<4, 4>>>,

  CCIfType<[i32], CCIfSubtarget<"isTargetWin64()", CCAssignToReg<[ECX, EDX, R8D, R9D]>>>,
  CCIfType<[i64], CCIfSubtarget<"isTargetWin64()", CCAssignToReg<[RCX, RDX, R8,  R9 ]>>>,

  CCIfType<[i32], CCAssignToReg<[EDI, ESI, EDX, ECX]>>,
  CCIfType<[i64], CCAssignToReg<[RDI, RSI, RDX, RCX]>>,

 // The SSE vector arguments are passed in XMM registers.
  CCIfType<[f32, f64, v4i32, v2i64, v4f32, v2f64],
           CCAssignToReg<[XMM0, XMM1, XMM2, XMM3]>>,
  
  // The 256-bit vector arguments are passed in YMM registers.
  CCIfType<[v8f32, v4f64, v8i32, v4i64],
                CCAssignToReg<[YMM0, YMM1, YMM2, YMM3]>>,
  
  CCIfSubtarget<"isTargetWin64()", CCDelegateTo<CC_X86_Win64_C>>,
  CCDelegateTo<CC_X86_64_C>
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

  // The first 3 __m64 vector arguments are passed in mmx registers if the
  // call is not a vararg call.
  CCIfNotVarArg<CCIfType<[x86mmx],
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

  // The first 4 AVX 256-bit vector arguments are passed in YMM registers.
  CCIfNotVarArg<CCIfType<[v32i8, v16i16, v8i32, v4i64, v8f32, v4f64],
                CCIfSubtarget<"hasAVX()",
                CCAssignToReg<[YMM0, YMM1, YMM2, YMM3]>>>>,

  // Other SSE vectors get 16-byte stack slots that are 16-byte aligned.
  CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64], CCAssignToStack<16, 16>>,

  // 256-bit AVX vectors get 32-byte stack slots that are 32-byte aligned.
  CCIfType<[v32i8, v16i16, v8i32, v4i64, v8f32, v4f64],
           CCAssignToStack<32, 32>>,

  // __m64 vectors get 8-byte stack slots that are 4-byte aligned. They are
  // passed in the parameter area.
  CCIfType<[x86mmx], CCAssignToStack<8, 4>>]>;

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
  CCIfInReg<CCIfType<[i32], CCAssignToReg<[ECX, EDX]>>>,

  // Otherwise, same as everything else.
  CCDelegateTo<CC_X86_32_Common>
]>;

def CC_X86_32_ThisCall : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // Pass sret arguments indirectly through EAX
  CCIfSRet<CCAssignToReg<[EAX]>>,

  // The first integer argument is passed in ECX
  CCIfType<[i32], CCAssignToReg<[ECX]>>,

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

//===----------------------------------------------------------------------===//
// X86 Root Argument Calling Conventions
//===----------------------------------------------------------------------===//

// This is the root argument convention for the X86-32 backend.
def CC_X86_32 : CallingConv<[
  CCIfCC<"CallingConv::X86_FastCall", CCDelegateTo<CC_X86_32_FastCall>>,
  CCIfCC<"CallingConv::X86_ThisCall", CCDelegateTo<CC_X86_32_ThisCall>>,
  CCIfCC<"CallingConv::Fast", CCDelegateTo<CC_X86_32_FastCC>>,
  CCIfCC<"CallingConv::GHC", CCDelegateTo<CC_X86_32_GHC>>,

  // Otherwise, drop to normal X86-32 CC
  CCDelegateTo<CC_X86_32_C>
]>;

// This is the root argument convention for the X86-64 backend.
def CC_X86_64 : CallingConv<[
  CCIfCC<"CallingConv::GHC", CCDelegateTo<CC_X86_64_GHC>>,

  // Mingw64 and native Win64 use Win64 CC
  CCIfSubtarget<"isTargetWin64()", CCDelegateTo<CC_X86_Win64_C>>,

  // Otherwise, drop to normal X86-64 CC
  CCDelegateTo<CC_X86_64_C>
]>;

// This is the argument convention used for the entire X86 backend.
def CC_X86 : CallingConv<[
  CCIfCC<"CallingConv::Intel_OCL_BI", CCDelegateTo<CC_Intel_OCL_BI>>,
  CCIfSubtarget<"is64Bit()", CCDelegateTo<CC_X86_64>>,
  CCDelegateTo<CC_X86_32>
]>;

//===----------------------------------------------------------------------===//
// Callee-saved Registers.
//===----------------------------------------------------------------------===//

def CSR_NoRegs : CalleeSavedRegs<(add)>;

def CSR_32 : CalleeSavedRegs<(add ESI, EDI, EBX, EBP)>;
def CSR_64 : CalleeSavedRegs<(add RBX, R12, R13, R14, R15, RBP)>;

def CSR_32EHRet : CalleeSavedRegs<(add EAX, EDX, CSR_32)>;
def CSR_64EHRet : CalleeSavedRegs<(add RAX, RDX, CSR_64)>;

def CSR_Win64 : CalleeSavedRegs<(add RBX, RBP, RDI, RSI, R12, R13, R14, R15,
                                     (sequence "XMM%u", 6, 15))>;


// Standard C + YMM6-15
def CSR_Win64_Intel_OCL_BI_AVX : CalleeSavedRegs<(add RBX, RBP, RDI, RSI, R12,
                                                  R13, R14, R15, 
                                                  (sequence "YMM%u", 6, 15))>;

//Standard C + XMM 8-15
def CSR_64_Intel_OCL_BI       : CalleeSavedRegs<(add CSR_64,
                                                 (sequence "XMM%u", 8, 15))>;

//Standard C + YMM 8-15
def CSR_64_Intel_OCL_BI_AVX    : CalleeSavedRegs<(add CSR_64,
                                                  (sequence "YMM%u", 8, 15))>;
