; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O0 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 < %s | FileCheck %s

; Verify that returning multiple return values in registers works,
; both with fast-isel and regular isel.

define { i32, i32, i32, i32 } @foo() nounwind {
  %A1 = insertvalue { i32, i32, i32, i32 } undef, i32 1, 0
  %A2 = insertvalue { i32, i32, i32, i32 } %A1, i32 2, 1
  %A3 = insertvalue { i32, i32, i32, i32 } %A2, i32 3, 2
  %A4 = insertvalue { i32, i32, i32, i32 } %A3, i32 4, 3
  ret { i32, i32, i32, i32 } %A4
}

; CHECK-LABEL: foo:
; CHECK: li 3, 1
; CHECK: li 4, 2
; CHECK: li 5, 3
; CHECK: li 6, 4
; CHECK: blr

