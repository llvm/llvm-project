; RUN: opt -S -passes=instcombine -verify-each < %s | FileCheck %s

; Inversion preserves the observed block/successor events and swaps weights.
; CHECK-LABEL: define i32 @invert(
; CHECK: br i1 %cond, label %no, label %yes, !prof ![[W:[0-9]+]], !block.uniformity.profile ![[U:[0-9]+]], !branch.uniformity.profile ![[U]]{{$}}
; CHECK: ![[W]] = !{!"branch_weights", i32 10, i32 90}

define i32 @invert(i1 %cond) !uniformity.profile !0 {
entry:
  %neg = xor i1 %cond, true
  br i1 %neg, label %yes, label %no, !prof !1, !block.uniformity.profile !0, !branch.uniformity.profile !0
yes:
  ret i32 1
no:
  ret i32 0
}
!0 = !{}
!1 = !{!"branch_weights", i32 90, i32 10}
