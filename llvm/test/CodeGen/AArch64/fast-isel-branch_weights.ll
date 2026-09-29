; RUN: llc -mtriple=arm64-apple-darwin -aarch64-enable-atomic-cfg-tidy=0                               -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin -aarch64-enable-atomic-cfg-tidy=0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

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
