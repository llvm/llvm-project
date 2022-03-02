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

multiclass VectorStore<ValueType DataVT,
    ValueType PtrVT, ValueType MaskVT,
    string STWithMask, string STNoMask> {
  // Unmasked (imm stride).
  def : Pat<(vvp_store
               DataVT:$val, PtrVT:$addr,
               (i64 simm7:$stride), (MaskVT true_mask), i32:$avl),
            (!cast<Instruction>(STNoMask#"irvl")
               (LO7 $stride), $addr, $val, $avl)>;
  // Unmasked.
  def : Pat<(vvp_store
               DataVT:$val, PtrVT:$addr,
               i64:$stride, (MaskVT true_mask), i32:$avl),
            (!cast<Instruction>(STNoMask#"rrvl")
               $stride, $addr, $val, $avl)>;
  // Masked (imm stride).
  def : Pat<(vvp_store
               DataVT:$val, PtrVT:$addr,
               (i64 simm7:$stride), MaskVT:$mask, i32:$avl),
            (!cast<Instruction>(STWithMask#"irvml")
               (LO7 $stride), $addr, $val, $mask, $avl)>;
  // Masked.
  def : Pat<(vvp_store
               DataVT:$val, PtrVT:$addr,
               i64:$stride, MaskVT:$mask, i32:$avl),
            (!cast<Instruction>(STWithMask#"rrvml")
               $stride, $addr, $val, $mask, $avl)>;
}

defm : VectorStore<v256f64, i64, v256i1, "VST",  "VST">;
defm : VectorStore<v256i64, i64, v256i1, "VST",  "VST">;
defm : VectorStore<v256f32, i64, v256i1, "VSTU", "VSTU">;
defm : VectorStore<v256i32, i64, v256i1, "VSTL", "VSTL">;

multiclass VectorLoad<ValueType DataVT,
    ValueType PtrVT, ValueType MaskVT,
    string GTWithMask, string LDNoMask> {
  // Unmasked (imm stride).
  def : Pat<(DataVT (vvp_load
               PtrVT:$addr, (i64 simm7:$stride),
               (MaskVT true_mask), i32:$avl)),
            (!cast<Instruction>(LDNoMask#"irl")
               (LO7 $stride), $addr, $avl)>;
  // Unmasked.
  def : Pat<(DataVT (vvp_load
               PtrVT:$addr, i64:$stride,
               (MaskVT true_mask), i32:$avl)),
            (!cast<Instruction>(LDNoMask#"rrl")
               $stride, PtrVT:$addr, $avl)>;
  // Masked (imm stride).
  def : Pat<(DataVT (vvp_load
               PtrVT:$addr, (i64 simm7:$stride),
               MaskVT:$mask, i32:$avl)),
            (!cast<Instruction>(GTWithMask#"vizml")
               (VADDULrvml $addr,
                  (VMULULivml (LO7 $stride), (VSEQl $avl), $mask, $avl),
                  $mask, $avl),
               0, 0,
               $mask,
               $avl)>;
  // Masked.
  def : Pat<(DataVT (vvp_load
               PtrVT:$addr, i64:$stride, MaskVT:$mask, i32:$avl)),
            (!cast<Instruction>(GTWithMask#"vizml")
               (VADDULrvml $addr,
                  (VMULULrvml $stride, (VSEQl $avl), $mask, $avl),
                  $mask, $avl),
               0, 0,
               $mask,
               $avl)>;
}

defm : VectorLoad<v256f64, i64, v256i1, "VGT",    "VLD">;
defm : VectorLoad<v256i64, i64, v256i1, "VGT",    "VLD">;
defm : VectorLoad<v256f32, i64, v256i1, "VGTU",   "VLDU">;
defm : VectorLoad<v256i32, i64, v256i1, "VGTLZX", "VLDLZX">;



multiclass Binary_rv<SDPatternOperator OpNode,
    ValueType ScalarVT, ValueType DataVT,
    ValueType MaskVT, string OpBaseName> {
  // Masked with passthru, broadcast.
  def : Pat<(vvp_select
                (OpNode
                    (any_broadcast ScalarVT:$sx),
                    DataVT:$vy,
                    (MaskVT srcvalue),
                    (i32 srcvalue)),
                DataVT:$vfalse,
                MaskVT:$mask,
                i32:$pivot),
            (!cast<Instruction>(OpBaseName#"rvml_v")
                ScalarVT:$sx,
                $vy,
                $mask,
                $pivot,
                $vfalse)>;

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
  // Masked with passthru, broadcast.
  def : Pat<(vvp_select
                (OpNode
                    DataVT:$vx,
                    (any_broadcast ScalarVT:$sy),
                    (MaskVT srcvalue),
                    (i32 srcvalue)),
                DataVT:$vfalse,
                MaskVT:$mask,
                i32:$pivot),
            (!cast<Instruction>(OpBaseName#"vrml_v")
                $vx,
                ScalarVT:$sy,
                $mask,
                $pivot,
                $vfalse)>;

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
  // Masked with passthru, broadcast.
  def : Pat<(vvp_select
                (OpNode
                    DataVT:$vx,
                    DataVT:$vy,
                    (MaskVT srcvalue),
                    (i32 srcvalue)),
                DataVT:$vfalse,
                MaskVT:$mask,
                i32:$pivot),
            (!cast<Instruction>(OpBaseName#"vvml_v")
                $vx,
                $vy,
                $mask,
                $pivot,
                $vfalse)>;

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

defm : Binary_rv_vv<c_vvp_and,
                    i64, v512i32, v512i1, "PVAND">;
defm : Binary_rv_vv<c_vvp_or,
                    i64, v512i32, v512i1, "PVOR">;
defm : Binary_rv_vv<c_vvp_xor,
                    i64, v512i32, v512i1, "PVXOR">;

defm : Binary_rv_vv<c_vvp_add,
                    i64, v512i32, v512i1, "PVADDU">;
defm : Binary_rv_vv<vvp_sub,
                    i64, v512i32, v512i1, "PVSUBU">;
defm : Binary_vr_vv<vvp_srl,
                    i64, v512i32, v512i1, "PVSRL">;
defm : Binary_vr_vv<vvp_sra,
                    i64, v512i32, v512i1, "PVSRA">;
defm : Binary_vr_vv<vvp_shl,
                    i64, v512i32, v512i1, "PVSLL">;

defm : Binary_rv_vv<c_vvp_fadd,
                    i64, v512f32, v512i1, "PVFADD">;
defm : Binary_rv_vv<c_vvp_fmul,
                    i64, v512f32, v512i1, "PVFMUL">;
defm : Binary_rv_vv<vvp_fsub,
                    i64, v512f32, v512i1, "PVFSUB">;

multiclass Merge_mvv<
    SDPatternOperator OpNode,
    ValueType DataVT, ValueType MaskVT,
    string OpBaseName> {
  // Masked.
  def : Pat<(OpNode
                DataVT:$vtrue, DataVT:$vfalse,
                MaskVT:$vm,
                i32:$avl),
            (!cast<Instruction>(OpBaseName#"vvml_v")
                $vfalse, $vtrue, $vm, $avl, $vfalse)>;
}

multiclass Merge_mvv_ShortLong<
    SDPatternOperator OpNode,
    ValueType LongDataVT, ValueType ShortDataVT,
    string OpBaseName> {
  defm : Merge_mvv<OpNode,
                   LongDataVT, v256i1,
                   OpBaseName>;
  defm : Merge_mvv<OpNode,
                   ShortDataVT, v256i1,
                   OpBaseName>;
}

defm : Merge_mvv_ShortLong<vvp_select,
                           v256f64,
                           v256f32, "VMRG">;
defm : Merge_mvv_ShortLong<vvp_select,
                           v256i64,
                           v256i32, "VMRG">;

multiclass Set_CC<ValueType DataVT, string FmkBaseName, string CmpBaseName, SDPatternOperator CCMatcher, SDNodeXForm CCConv> {
  // Unmasked.
  def : Pat<(v256i1 (vvp_setcc
              DataVT:$LHS, DataVT:$RHS, CCMatcher:$cond, (v256i1 true_mask), i32:$vl)),
              (!cast<Instruction>(FmkBaseName#"vl")
                (CCConv $cond),
                (!cast<Instruction>(CmpBaseName#"vvl")
                  $LHS, $RHS, $vl),
                $vl)>;
  // Masked.
  def : Pat<(v256i1 (vvp_setcc
              DataVT:$LHS, DataVT:$RHS, CCMatcher:$cond, v256i1:$vm, i32:$vl)),
              (!cast<Instruction>(FmkBaseName#"vml")
                (CCConv $cond),
                (!cast<Instruction>(CmpBaseName#"vvl")
                  $LHS, $RHS, $vl),
                $vm, $vl)>;
}

defm : Set_CC<v256i64,"VFMKL","VCMPUL",CCUIOp,icond2cc>;
defm : Set_CC<v256i64,"VFMKL","VCMPSL",CCSIOp,icond2cc>;
defm : Set_CC<v256f64,"VFMKL","VFCMPD",cond,fcond2cc>;

defm : Set_CC<v256i32,"VFMKW","VCMPUW",CCUIOp,icond2cc>;
defm : Set_CC<v256i32,"VFMKW","VCMPSWZX",CCSIOp,icond2cc>;
defm : Set_CC<v256f32,"VFMKS","VFCMPS",cond,fcond2cc>;
