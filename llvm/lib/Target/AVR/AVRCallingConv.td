//===-- AVRCallingConv.td - Calling Conventions for AVR ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for AVR architecture.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AVR Return Value Calling Convention
//===----------------------------------------------------------------------===//

def RetCC_AVR : CallingConv
<[
  // i8 is returned in R24.
  CCIfType<[i8], CCAssignToReg<[R24]>>,

  // i16 are returned in R25:R24, R23:R22, R21:R20 and R19:R18.
  CCIfType<[i16], CCAssignToReg<[R25R24, R23R22, R21R20, R19R18]>>
]>;

// Special return value calling convention for runtime functions.
def RetCC_AVR_BUILTIN : CallingConv
<[
  CCIfType<[i8], CCAssignToReg<[R24,R25]>>,
  CCIfType<[i16], CCAssignToReg<[R23R22, R25R24]>>
]>;

//===----------------------------------------------------------------------===//
// AVR Argument Calling Conventions
//===----------------------------------------------------------------------===//

// The calling conventions are implemented in custom C++ code

// Calling convention for variadic functions.
def ArgCC_AVR_Vararg : CallingConv
<[
  // i16 are always passed through the stack with an alignment of 1.
  CCAssignToStack<2, 1>
]>;

// Special argument calling convention for
// division runtime functions.
def ArgCC_AVR_BUILTIN_DIV : CallingConv
<[
  CCIfType<[i8], CCAssignToReg<[R24,R22]>>,
  CCIfType<[i16], CCAssignToReg<[R25R24, R23R22]>>
]>;

//===----------------------------------------------------------------------===//
// Callee-saved register lists.
//===----------------------------------------------------------------------===//

def CSR_Normal : CalleeSavedRegs<(add R29, R28, (sequence "R%u", 17, 2))>;
def CSR_Interrupts : CalleeSavedRegs<(add (sequence "R%u", 31, 0))>;
