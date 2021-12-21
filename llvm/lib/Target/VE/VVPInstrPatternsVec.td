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

multiclass Binary_vr<SDPatternOperator OpNode,
    ValueType ScalarVT, ValueType DataVT,
    ValueType MaskVT, string OpBaseName> {
  // Masked with select, broadcast.
  // TODO

  // Unmasked, broadcast.
  def : Pat<(OpNode
                DataVT:$vx, (any_broadcast ScalarVT:$sy),
                (MaskVT true_mask),
                i32:$avl),
            (!cast<Instruction>(OpBaseName#"vrl")
                $vx, ScalarVT:$sy, $avl)>;
  // Masked, broadcast.
  def : Pat<(OpNode
                DataVT:$vx, (any_broadcast ScalarVT:$sy),
                MaskVT:$mask,
                i32:$avl),
            (!cast<Instruction>(OpBaseName#"vrml")
                $vx, ScalarVT:$sy, $mask, $avl)>;
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

multiclass Binary_vr_vv<
    SDPatternOperator OpNode,
    ValueType ScalarVT, ValueType DataVT, ValueType MaskVT,
    string OpBaseName> {
  defm : Binary_vr<OpNode, ScalarVT, DataVT, MaskVT, OpBaseName>;
  defm : Binary_vv<OpNode, DataVT, MaskVT, OpBaseName>;
}

multiclass Binary_rv_vr_vv<
    SDPatternOperator OpNode,
    ValueType ScalarVT, ValueType DataVT, ValueType MaskVT,
    string OpBaseName> {
  defm : Binary_rv<OpNode, ScalarVT, DataVT, MaskVT, OpBaseName>;
  defm : Binary_vr_vv<OpNode, ScalarVT, DataVT, MaskVT, OpBaseName>;
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

multiclass Binary_vr_vv_ShortLong<
    SDPatternOperator OpNode,
    ValueType LongScalarVT, ValueType LongDataVT, string LongOpBaseName,
    ValueType ShortScalarVT, ValueType ShortDataVT, string ShortOpBaseName> {
  defm : Binary_vr_vv<OpNode,
                      LongScalarVT, LongDataVT, v256i1,
                      LongOpBaseName>;
  defm : Binary_vr_vv<OpNode,
                      ShortScalarVT, ShortDataVT, v256i1,
                      ShortOpBaseName>;
}

multiclass Binary_rv_vr_vv_ShortLong<
    SDPatternOperator OpNode,
    ValueType LongScalarVT, ValueType LongDataVT, string LongOpBaseName,
    ValueType ShortScalarVT, ValueType ShortDataVT, string ShortOpBaseName> {
  defm : Binary_rv_vr_vv<OpNode,
                      LongScalarVT, LongDataVT, v256i1,
                      LongOpBaseName>;
  defm : Binary_rv_vr_vv<OpNode,
                      ShortScalarVT, ShortDataVT, v256i1,
                      ShortOpBaseName>;
}

defm : Binary_rv_vv_ShortLong<c_vvp_add,
                              i64, v256i64, "VADDSL",
                              i32, v256i32, "VADDSWSX">;
defm : Binary_rv_vv_ShortLong<vvp_sub,
                              i64, v256i64, "VSUBSL",
                              i32, v256i32, "VSUBSWSX">;
defm : Binary_rv_vv_ShortLong<c_vvp_mul,
                              i64, v256i64, "VMULSL",
                              i32, v256i32, "VMULSWSX">;
defm : Binary_rv_vr_vv_ShortLong<vvp_sdiv,
                              i64, v256i64, "VDIVSL",
                              i32, v256i32, "VDIVSWSX">;
defm : Binary_rv_vr_vv_ShortLong<vvp_udiv,
                              i64, v256i64, "VDIVUL",
                              i32, v256i32, "VDIVUW">;
defm : Binary_rv_vv_ShortLong<c_vvp_and,
                              i64, v256i64, "VAND",
                              i32, v256i32, "PVANDLO">;
defm : Binary_rv_vv_ShortLong<c_vvp_or,
                              i64, v256i64, "VOR",
                              i32, v256i32, "PVORLO">;
defm : Binary_rv_vv_ShortLong<c_vvp_xor,
                              i64, v256i64, "VXOR",
                              i32, v256i32, "PVXORLO">;
defm : Binary_vr_vv_ShortLong<vvp_shl,
                              i64, v256i64, "VSLL",
                              i32, v256i32, "PVSLLLO">;
defm : Binary_vr_vv_ShortLong<vvp_sra,
                              i64, v256i64, "VSRAL",
                              i32, v256i32, "PVSRALO">;
defm : Binary_vr_vv_ShortLong<vvp_srl,
                              i64, v256i64, "VSRL",
                              i32, v256i32, "PVSRLLO">;

defm : Binary_rv_vv_ShortLong<c_vvp_fadd,
                              f64, v256f64, "VFADDD",
                              f32, v256f32, "PVFADDUP">;
defm : Binary_rv_vv_ShortLong<c_vvp_fmul,
                              f64, v256f64, "VFMULD",
                              f32, v256f32, "PVFMULUP">;
defm : Binary_rv_vv_ShortLong<vvp_fsub,
                              f64, v256f64, "VFSUBD",
                              f32, v256f32, "PVFSUBUP">;
defm : Binary_rv_vr_vv_ShortLong<vvp_fdiv,
                              f64, v256f64, "VFDIVD",
                              f32, v256f32, "VFDIVS">;
