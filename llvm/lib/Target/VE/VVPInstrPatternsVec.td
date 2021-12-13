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

multiclass Binary_rv<SDPatternOperator OpNode,
    ValueType ScalarVT, ValueType DataVT,
    ValueType MaskVT, string OpBaseName> {
  // Masked with select, broadcast.
  // TODO

  // Unmasked, broadcast.
  def : Pat<(OpNode
                (any_broadcast ScalarVT:$sx), DataVT:$vy,
                (MaskVT true_mask),
                i32:$avl),
            (!cast<Instruction>(OpBaseName#"rvl")
                ScalarVT:$sx, $vy, $avl)>;
  // Masked, broadcast.
  def : Pat<(OpNode
                (any_broadcast ScalarVT:$sx), DataVT:$vy,
                MaskVT:$mask,
                i32:$avl),
            (!cast<Instruction>(OpBaseName#"rvml")
                ScalarVT:$sx, $vy, $mask, $avl)>;
}

multiclass Binary_vv<SDPatternOperator OpNode,
    ValueType DataVT,
    ValueType MaskVT, string OpBaseName> {
  // Masked with select.
  // TODO

  // Unmasked.
  def : Pat<(OpNode
                DataVT:$vx, DataVT:$vy,
                (MaskVT true_mask),
                i32:$avl),
            (!cast<Instruction>(OpBaseName#"vvl")
                $vx, $vy, $avl)>;

  // Masked.
  def : Pat<(OpNode
                DataVT:$vx, DataVT:$vy,
                MaskVT:$mask,
                i32:$avl),
            (!cast<Instruction>(OpBaseName#"vvml")
                $vx, $vy, $mask, $avl)>;
}

multiclass Binary_rv_vv<
    SDPatternOperator OpNode,
    ValueType ScalarVT, ValueType DataVT, ValueType MaskVT,
    string OpBaseName> {
  defm : Binary_rv<OpNode, ScalarVT, DataVT, MaskVT, OpBaseName>;
  defm : Binary_vv<OpNode, DataVT, MaskVT, OpBaseName>;
}

// Expand both 64bit and 32 bit variant (256 elements)
multiclass Binary_rv_vv_ShortLong<
    SDPatternOperator OpNode,
    ValueType LongScalarVT, ValueType LongDataVT, string LongOpBaseName,
    ValueType ShortScalarVT, ValueType ShortDataVT, string ShortOpBaseName> {
  defm : Binary_rv_vv<OpNode,
                      LongScalarVT, LongDataVT, v256i1,
                      LongOpBaseName>;
  defm : Binary_rv_vv<OpNode,
                      ShortScalarVT, ShortDataVT, v256i1,
                      ShortOpBaseName>;
}


defm : Binary_rv_vv_ShortLong<c_vvp_add,
                              i64, v256i64, "VADDSL",
                              i32, v256i32, "VADDSWSX">;
defm : Binary_rv_vv_ShortLong<c_vvp_and,
                              i64, v256i64, "VAND",
                              i32, v256i32, "PVANDLO">;
