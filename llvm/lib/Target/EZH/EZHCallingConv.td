//===-- EZHCallingConv.td - Calling Conventions for EZH --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// CCIfAlign - Match of the original alignment of the arg
class CCIfAlign<string Align, CCAction A>:
  CCIf<!strconcat("ArgFlags.getNonZeroOrigAlign() >= ", Align), A>;

def CC_EZH : CallingConv<[
  CCIfByVal<CCPassByVal<4, 4>>,

  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // i64/f64 (or >= 8-byte aligned types) are passed in even pairs of GPRs (R0/R1 or R2/R3)
  // We match >= 8-alignment i32 parts, shadowing R1 as a pad register!
  CCIfType<[i32], CCIfAlign<"8", CCAssignToRegWithShadow<[R0, R2], [R0, R1]>>>,
  CCIfType<[i32], CCIf<"ArgFlags.getNonZeroOrigAlign() < Align(8)",
                       CCAssignToReg<[R0, R1, R2, R3]>>>,

  // Stack allocations with strict 8-byte alignment support for spilled types!
  CCIfType<[i32], CCIfAlign<"8", CCAssignToStackWithShadow<4, 8, [R0, R1, R2, R3]>>>,
  CCIfType<[i32], CCAssignToStackWithShadow<4, 4, [R0, R1, R2, R3]>>
]>;

def RetCC_EZH : CallingConv<[
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // Return in R0, R1, R2, R3
  CCIfType<[i32, f32], CCAssignToReg<[R0, R1, R2, R3]>>
]>;

def CSR_EZH : CalleeSavedRegs<(add RA, R4, R5, R6, R7)>;
