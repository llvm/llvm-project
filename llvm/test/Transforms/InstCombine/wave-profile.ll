; RUN: opt -S -passes='instcombine,verify' %s | FileCheck %s

; InstCombine canonicalizes a negated branch condition by swapping successors.
; The stable successor identities must follow that equivalent edge swap.

define void @_Z14divergent_loopPVi(i1 %condition) !wave.profile !0 {
; CHECK-LABEL: define void @_Z14divergent_loopPVi(
; CHECK: entry:
; CHECK-NEXT: br i1 %condition, label %right, label %left, !wave.profile.block [[ENTRY:![0-9]+]]
entry:
  %not = xor i1 %condition, true
  br i1 %not, label %left, label %right, !wave.profile.block !1

left:
  ret void, !wave.profile.block !2

right:
  ret void, !wave.profile.block !3
}

; CHECK: [[ENTRY]] = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 2, i64 1}

!0 = !{i64 2, i64 2480672276464841217, i64 100, i64 10, i64 90}
!1 = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1, i64 2}
!2 = !{i64 2, i64 2480672276464841217, i64 1, i64 1}
!3 = !{i64 2, i64 2480672276464841217, i64 2, i64 1}
