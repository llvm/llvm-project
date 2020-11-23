//===-- VEInstrPatternsVec.td - VEC_-type SDNodes and isel for VE Target --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the VEC_* prefixed intermediate SDNodes and their
// isel patterns.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

multiclass vbrd_elem32<ValueType v32, ValueType s32, SDPatternOperator ImmOp, SDNodeXForm ImmCast, int SubRegIdx> {
  // VBRDil
  def : Pat<(v32 (vec_broadcast (s32 ImmOp:$sy), i32:$vl)),
            (VBRDil (ImmCast $sy), i32:$vl)>;

  // VBRDrl
  def : Pat<(v32 (vec_broadcast s32:$sy, i32:$vl)),
            (VBRDrl
              (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $sy, SubRegIdx),
              i32:$vl)>;
}

defm : vbrd_elem32<v256f32, f32, simm7fp, LO7FP, sub_f32>;
defm : vbrd_elem32<v256i32, i32, simm7, LO7, sub_i32>;

multiclass vbrd_elem64<ValueType v64, ValueType s64, SDPatternOperator ImmOp, SDNodeXForm ImmCast> {
  // VBRDil
  def : Pat<(v64 (vec_broadcast (s64 ImmOp:$sy), i32:$vl)),
            (VBRDil (ImmCast $sy), i32:$vl)>;

  // VBRDrl
  def : Pat<(v64 (vec_broadcast s64:$sy, i32:$vl)),
            (VBRDrl s64:$sy, i32:$vl)>;
}

defm : vbrd_elem64<v256f64, f64, simm7fp, LO7FP>;
defm : vbrd_elem64<v256i64, i64, simm7, LO7>;
