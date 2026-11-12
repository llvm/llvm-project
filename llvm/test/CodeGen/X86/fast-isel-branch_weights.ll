; RUN: llc < %s                             -mtriple=x86_64-apple-darwin10 | FileCheck %s
; RUN: llc < %s -fast-isel -fast-isel-abort=1 -mtriple=x86_64-apple-darwin10 | FileCheck %s

; Test if the BBs are reordred according to their branch weights.
define i64 @branch_weights_test(i64 %a, i64 %b) {
; CHECK-LABEL: branch_weights_test
; CHECK-LABEL: success
; CHECK-LABEL: fail
  %1 = icmp ult i64 %a, %b
  br i1 %1, label %fail, label %success, !prof !0

fail:
  ret i64 -1

success:
  ret i64 0
}

!0 = !{!"branch_weights", i32 0, i32 2147483647}
