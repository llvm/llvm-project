; RUN: opt < %s -passes='require<profile-summary>,function(chr)' -S | FileCheck %s

; Without uniformity profile, Boolean selects retain the existing CHR behavior.
define i1 @boolean_no_uniformity(i1 %condition, i1 %second_condition, i1 %left, i1 %right) !prof !14 {
; CHECK-LABEL: define i1 @boolean_no_uniformity(
; CHECK: entry.split.nonchr:
entry:
  %first = select i1 %condition, i1 %left, i1 false, !prof !15
  %second = select i1 %second_condition, i1 %right, i1 false, !prof !15
  %result = and i1 %first, %second
  ret i1 %result
}

; Strong true bias does not enable Boolean-select CHR with uniformity profile.
define i1 @boolean_true_bias(i1 %condition, i1 %second_condition, i1 %left, i1 %right) !prof !14 !uniformity.profile !16 {
; CHECK-LABEL: define i1 @boolean_true_bias(
; CHECK-NOT: br i1
; CHECK: %first = select i1 %condition, i1 %left, i1 false
; CHECK: %second = select i1 %second_condition, i1 %right, i1 false
; CHECK-NOT: br i1
; CHECK: ret i1 %result
entry:
  %first = select i1 %condition, i1 %left, i1 false, !prof !15
  %second = select i1 %second_condition, i1 %right, i1 false, !prof !15
  %result = and i1 %first, %second
  ret i1 %result
}

; Check the false-biased candidate set as well.
define i1 @boolean_false_bias(i1 %condition, i1 %second_condition, i1 %left, i1 %right) !prof !14 !uniformity.profile !16 {
; CHECK-LABEL: define i1 @boolean_false_bias(
; CHECK-NOT: br i1
; CHECK: %first = select i1 %condition, i1 true, i1 %left
; CHECK: %second = select i1 %second_condition, i1 true, i1 %right
; CHECK-NOT: br i1
; CHECK: ret i1 %result
entry:
  %first = select i1 %condition, i1 true, i1 %left, !prof !17
  %second = select i1 %second_condition, i1 true, i1 %right, !prof !17
  %result = and i1 %first, %second
  ret i1 %result
}

; Numerical selects remain eligible with the same function-level marker.
define half @half_selects(i1 %first_condition, i1 %second_condition, half %value) !prof !14 !uniformity.profile !16 {
; CHECK-LABEL: define half @half_selects(
; CHECK: entry.split.nonchr:
entry:
  %first = select i1 %first_condition, half %value, half 0.0, !prof !15
  %second = select i1 %second_condition, half %first, half 0.0, !prof !15
  ret half %second
}

; A Boolean candidate rejects its whole scope, including numerical candidates.
define half @mixed_selects(i1 %condition, i1 %valid, i1 %numeric_condition, half %value, ptr %out) !prof !14 !uniformity.profile !16 {
; CHECK-LABEL: define half @mixed_selects(
; CHECK-NOT: br i1
; CHECK: %first = select i1 %condition, i1 %valid, i1 false
; CHECK: %second = select i1 %numeric_condition, half %value, half 0.000000e+00
; CHECK-NOT: br i1
; CHECK: ret half %second
entry:
  %first = select i1 %condition, i1 %valid, i1 false, !prof !15
  %second = select i1 %numeric_condition, half %value, half 0.0, !prof !15
  store i1 %first, ptr %out
  ret half %second
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 1}
!8 = !{!"NumFunctions", i64 1}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11}
!11 = !{i32 999999, i64 1, i32 1}
!14 = !{!"function_entry_count", i64 100}
!15 = !{!"branch_weights", i32 1000, i32 0}
!16 = !{}
!17 = !{!"branch_weights", i32 0, i32 1000}
