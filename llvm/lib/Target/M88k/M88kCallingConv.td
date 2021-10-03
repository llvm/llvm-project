//===-- M88kCallingConv.td - M88k Calling Conventions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the M88k architectures.
// Only the OpenBSD variant is supported.
//
//===----------------------------------------------------------------------===//

// M88k 32-bit ELF C Calling convention.
def CC_M88k : CallingConv<[
  // Promote i8/i16 args to i32.
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // Register R12 is used to pass structure return pointer.
  CCIfSRet<CCIfType<[i32], CCAssignToReg<[R12]>>>,

  // Registers R2 to R9 are used for passing parameters.
  CCIfType<[i32,f32], CCAssignToReg<[R2, R3, R4, R5, R6, R7, R8, R9]>>,

  // i64 and f64 values are in register pairs.
  CCIfType<[i64,f64], CCAssignToReg<[R2_R3, R4_R5, R6_R7, R8_R9]>>,

  // Other arguments are passed on the stack, at least 4-byte-aligned.
  CCAssignToStack<4, 4>
]>;

// M88k 32-bit ELF C return-value convention.
def RetCC_M88k : CallingConv<[
  // 32-bit values are returned in R2, 64-bit values in pair R2/R3.
  CCIfType<[i32,f32], CCAssignToReg<[R2,R3]>>,
  CCIfType<[i64,f64], CCAssignToReg<[R2_R3]>>
]>;

// M88k 32-bit ELF C callee saved registers.
def CSR_M88k : CalleeSavedRegs<(add (sequence "R%d", 14, 25), R1, R30)>;
