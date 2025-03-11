; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s
;
; This test case is reduced from RangeConstraintManager.cpp in a ASan build.
; It crashes reduceLoadWidth in DAGCombiner.cpp. Preservation of range
; metdata must ensure that ConstantRange truncation is strictly smaller.

define i8 @_ZN12_GLOBAL__N_121SymbolicRangeInferrer19VisitBinaryOperatorILN5clang18BinaryOperatorKindE15EEENS2_4ento8RangeSetES5_S5_NS2_8QualTypeE(ptr %valptr) {
entry:
  %val = load i8, ptr %valptr, align 4, !range !0, !noundef !1
  %retval.sroa.1.0.insert.ext.i = zext i8 %val to i64
  %retval.sroa.1.0.insert.shift.i = shl i64 %retval.sroa.1.0.insert.ext.i, 32
  %coerce.val.ii = trunc i64 %retval.sroa.1.0.insert.shift.i to i40
  store i40 %coerce.val.ii, ptr %valptr, align 4
  ret i8 %val
}

!0 = !{i8 0, i8 2}
!1 = !{}
