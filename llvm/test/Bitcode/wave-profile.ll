; RUN: llvm-as %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s
; RUN: opt -passes=verify -disable-output %t.bc
; RUN: verify-uselistorder %s

; Measured zero and an uninstrumented block have different validity flags.
; The serializer must retain both channels without treating wave counts as
; ordinary function-entry or branch-flow counts.

; CHECK-LABEL: define void @sparse(
; CHECK-SAME: !wave.profile [[PROFILE:![0-9]+]]
define void @sparse(i1 %condition) !wave.profile !0 {
entry:
  br i1 %condition, label %observed_zero, label %unmeasured, !wave.profile.block !1

observed_zero:
; CHECK: ret void, !wave.profile.block [[ZERO:![0-9]+]]
  ret void, !wave.profile.block !2

unmeasured:
; CHECK: ret void, !wave.profile.block [[UNMEASURED:![0-9]+]]
  ret void, !wave.profile.block !3
}

; Unknown versions and stale identities remain representable after transforms.
; CHECK-LABEL: define void @future(
; CHECK-SAME: !wave.profile [[FUTURE:![0-9]+]]
define void @future() !wave.profile !4 {
  ret void
}

; CHECK: [[PROFILE]] = !{i64 2, i64 123, i64 32, i64 0, i64 0}
; CHECK: [[ZERO]] = !{i64 2, i64 123, i64 1, i64 1}
; CHECK: [[UNMEASURED]] = !{i64 2, i64 123, i64 2, i64 0}
; CHECK: [[FUTURE]] = !{i64 3, i64 456, i64 0}
!0 = !{i64 2, i64 123, i64 32, i64 0, i64 0}
!1 = !{i64 2, i64 123, i64 0, i64 1, i64 1, i64 2}
!2 = !{i64 2, i64 123, i64 1, i64 1}
!3 = !{i64 2, i64 123, i64 2, i64 0}
!4 = !{i64 3, i64 456, i64 0}
