; RUN: llc < %s
;
; This test case is reduced from RangeConstraintManager.cpp in a ASan build.
; It crashes reduceLoadWidth in DAGCombiner.cpp. Preservation of range
; metdata must ensure that ConstantRange truncation is strictly smaller.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @_ZN12_GLOBAL__N_121SymbolicRangeInferrer19VisitBinaryOperatorILN5clang18BinaryOperatorKindE15EEENS2_4ento8RangeSetES5_S5_NS2_8QualTypeE() {
entry:
  %0 = load i8, ptr null, align 4, !range !0, !noundef !1
  %retval.sroa.1.0.insert.ext.i = zext i8 %0 to i64
  %retval.sroa.1.0.insert.shift.i = shl i64 %retval.sroa.1.0.insert.ext.i, 32
  %coerce.val.ii = trunc i64 %retval.sroa.1.0.insert.shift.i to i40
  store i40 %coerce.val.ii, ptr null, align 4
  ret ptr null
}

!0 = !{i8 0, i8 2}
!1 = !{}
