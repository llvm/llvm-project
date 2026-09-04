;; This file contains the previous semantic of the branch-target-enforcement, sign-return-address.
;; Used for test mixing a mixed link case and also verify the import too in llc.

; RUN: llc -mattr=+pauth -mattr=+bti %s -o - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare void @func()

define i32 @fiz_on() #0 {
entry:
  call void @func()
  ret i32 42
}

; CHECK-LABEL: fiz_on:
; CHECK:           paciasp
; CHECK:           bl func
; CHECK:           retaa

define i32 @fiz_off() #1 {
entry:
  ret i32 43
}

; CHECK-LABEL: fiz_off:
; CHECK-NOT:       pac
; CHECK-NOT:       hint
; CHECK-NOT:       bti
; CHECK:           ret

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { noinline nounwind optnone uwtable "branch-target-enforcement"="false" "sign-return-address"="none" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 0}
