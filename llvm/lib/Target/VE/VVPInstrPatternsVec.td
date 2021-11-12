//===----------- VVPInstrPatternsVec.td - VVP_* SDNode patterns -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes how VVP_* SDNodes are lowered to machine instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// VVP SDNode definitions.
//
//===----------------------------------------------------------------------===//
include "VVPInstrInfo.td"

multiclass VectorBinaryArith<
    SDPatternOperator OpNode,
    ValueType ScalarVT, ValueType DataVT, ValueType MaskVT,
    string OpBaseName> {
  // No mask.
  def : Pat<(OpNode
                (any_broadcast ScalarVT:$sx),
                DataVT:$vy, (MaskVT true_mask), i32:$avl),
            (!cast<Instruction>(OpBaseName#"rvl")
                ScalarVT:$sx, $vy, $avl)>;
  def : Pat<(OpNode DataVT:$vx, DataVT:$vy, (MaskVT true_mask), i32:$avl),
            (!cast<Instruction>(OpBaseName#"vvl")
                $vx, $vy, $avl)>;

  // Mask.
  def : Pat<(OpNode
                (any_broadcast ScalarVT:$sx),
                DataVT:$vy, MaskVT:$mask, i32:$avl),
            (!cast<Instruction>(OpBaseName#"rvml")
                ScalarVT:$sx, $vy, $mask, $avl)>;
  def : Pat<(OpNode DataVT:$vx, DataVT:$vy, MaskVT:$mask, i32:$avl),
            (!cast<Instruction>(OpBaseName#"vvml")
                $vx, $vy, $mask, $avl)>;

  // TODO We do not specify patterns for the immediate variants here. There
  // will be an immediate folding pass that takes care of switching to the
  // immediate variant where applicable.

  // TODO Fold vvp_select into passthru.
}

// Expand both 64bit and 32 bit variant (256 elements)
multiclass VectorBinaryArith_ShortLong<
    SDPatternOperator OpNode,
    ValueType LongScalarVT, ValueType LongDataVT, string LongOpBaseName,
    ValueType ShortScalarVT, ValueType ShortDataVT, string ShortOpBaseName> {
  defm : VectorBinaryArith<OpNode,
                           LongScalarVT, LongDataVT, v256i1,
                           LongOpBaseName>;
  defm : VectorBinaryArith<OpNode,
                           ShortScalarVT, ShortDataVT, v256i1,
                           ShortOpBaseName>;
}


defm : VectorBinaryArith_ShortLong<c_vvp_add,
                                   i64, v256i64, "VADDSL",
                                   i32, v256i32, "VADDSWSX">;
defm : VectorBinaryArith_ShortLong<c_vvp_and,
                                   i64, v256i64, "VAND",
                                   i32, v256i32, "PVANDLO">;
