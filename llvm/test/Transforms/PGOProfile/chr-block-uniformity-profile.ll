; RUN: opt < %s -passes='require<profile-summary>,function(chr,instcombine,simplifycfg)' -S | FileCheck %s

declare void @foo()

; Preserve the existing behavior when a function has no block-uniformity
; profile.
define void @no_profile(ptr %ptr) !prof !14 {
; CHECK-LABEL: define void @no_profile(
; CHECK: entry.split.nonchr:
entry:
  %value = load i32, ptr %ptr
  %bit0 = and i32 %value, 1
  %cond0 = icmp eq i32 %bit0, 0
  br i1 %cond0, label %bb1, label %bb0, !prof !15

bb0:
  call void @foo()
  br label %bb1

bb1:
  %bit1 = and i32 %value, 2
  %cond1 = icmp eq i32 %bit1, 0
  br i1 %cond1, label %exit, label %bb2, !prof !15

bb2:
  call void @foo()
  br label %exit

exit:
  ret void
}

; Uniform branches remain eligible for CHR when block-uniformity profile is
; present.
define void @uniform(ptr %ptr) !prof !14 !block.uniformity.profile !16 {
; CHECK-LABEL: define void @uniform(
; CHECK: entry.split.nonchr:
entry:
  %value = load i32, ptr %ptr
  %bit0 = and i32 %value, 1
  %cond0 = icmp eq i32 %bit0, 0
  br i1 %cond0, label %bb1, label %bb0, !prof !15,
      !branch.uniformity.profile !16

bb0:
  call void @foo()
  br label %bb1

bb1:
  %bit1 = and i32 %value, 2
  %cond1 = icmp eq i32 %bit1, 0
  br i1 %cond1, label %exit, label %bb2, !prof !15,
      !branch.uniformity.profile !16

bb2:
  call void @foo()
  br label %exit

exit:
  ret void
}

; Once the function has block-uniformity profile, a branch without branch
; uniformity metadata is divergent. Do not apply CHR to a scope containing one.
define void @divergent(ptr %ptr) !prof !14 !block.uniformity.profile !16 {
; CHECK-LABEL: define void @divergent(
; CHECK-NOT: split
; CHECK: ret void
entry:
  %value = load i32, ptr %ptr
  %bit0 = and i32 %value, 1
  %cond0 = icmp eq i32 %bit0, 0
  br i1 %cond0, label %bb1, label %bb0, !prof !15,
      !branch.uniformity.profile !16

bb0:
  call void @foo()
  br label %bb1

bb1:
  %bit1 = and i32 %value, 2
  %cond1 = icmp eq i32 %bit1, 0
  br i1 %cond1, label %exit, label %bb2, !prof !15

bb2:
  call void @foo()
  br label %exit

exit:
  ret void
}

; A divergent scope does not disable a separate uniform scope in the same
; function.
define void @per_scope(ptr %uniform_ptr, ptr %divergent_ptr) !prof !14 !block.uniformity.profile !16 {
; CHECK-LABEL: define void @per_scope(
; CHECK: entry.split.nonchr:
; CHECK-NOT: after.uniform.split.nonchr:
; CHECK: after.uniform:
entry:
  %uniform_value = load i32, ptr %uniform_ptr
  %uniform_bit0 = and i32 %uniform_value, 1
  %uniform_cond0 = icmp eq i32 %uniform_bit0, 0
  br i1 %uniform_cond0, label %uniform.bb1, label %uniform.bb0, !prof !15,
      !branch.uniformity.profile !16

uniform.bb0:
  call void @foo()
  br label %uniform.bb1

uniform.bb1:
  %uniform_bit1 = and i32 %uniform_value, 2
  %uniform_cond1 = icmp eq i32 %uniform_bit1, 0
  br i1 %uniform_cond1, label %after.uniform, label %uniform.bb2, !prof !15,
      !branch.uniformity.profile !16

uniform.bb2:
  call void @foo()
  br label %after.uniform

after.uniform:
  %divergent_value = load i32, ptr %divergent_ptr
  %divergent_bit0 = and i32 %divergent_value, 1
  %divergent_cond0 = icmp eq i32 %divergent_bit0, 0
  br i1 %divergent_cond0, label %divergent.bb1, label %divergent.bb0,
      !prof !15

divergent.bb0:
  call void @foo()
  br label %divergent.bb1

divergent.bb1:
  %divergent_bit1 = and i32 %divergent_value, 2
  %divergent_cond1 = icmp eq i32 %divergent_bit1, 0
  br i1 %divergent_cond1, label %exit, label %divergent.bb2, !prof !15

divergent.bb2:
  call void @foo()
  br label %exit

exit:
  ret void
}

; Select-only scopes are not described by branch-uniformity profile and keep
; their existing behavior.
define i32 @select_only(i32 %value) !prof !14 !block.uniformity.profile !16 {
; CHECK-LABEL: define i32 @select_only(
; CHECK: entry.split.nonchr:
entry:
  %bit0 = and i32 %value, 1
  %cond0 = icmp eq i32 %bit0, 0
  %sum0 = select i1 %cond0, i32 %value, i32 42, !prof !15
  %bit1 = and i32 %value, 2
  %cond1 = icmp eq i32 %bit1, 0
  %sum1 = select i1 %cond1, i32 %sum0, i32 43, !prof !15
  ret i32 %sum1
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
!15 = !{!"branch_weights", i32 0, i32 1}
!16 = !{}
