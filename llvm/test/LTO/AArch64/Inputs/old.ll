;; This file contains the previous semantic of the branch-target-enforcement, sign-return-address.
;; Used for test mixing a mixed link case and also verify the import too in llc.

; RUN: llc -mattr=+pauth -mattr=+bti %s -o - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define i32 @old_bti() #0 {
entry:
  ret i32 2
}

; CHECK-LABEL: old_bti:
; CHECK:           bti c
; CHECK:           mov
; CHECK:           ret

define i32 @old_pac() #1 {
entry:
  ret i32 2
}

; CHECK-LABEL: old_pac:
; CHECK:           paciasp
; CHECK:           mov
; CHECK:           retaa


define i32 @old_none() #2 {
entry:
  ret i32 3
}

; CHECK-LABEL: old_none:
; CHECK-NOT:           hint
; CHECK-NOT:           paci
; CHECK-NOT:           bti
; CHECK:           ret

declare i32 @func(i32)

define i32 @old_none_leaf() #3 {
entry:
  %0 = call i32 @func()
  ret i32 %0
}

; CHECK-LABEL: old_none_leaf:
; CHECK:           paciasp
; CHECK:           bl      func
; CHECK:           retaa

attributes #0 = { noinline nounwind optnone "branch-target-enforcement"="true" }
attributes #1 = { noinline nounwind optnone "branch-target-enforcement"="false" "sign-return-address"="all" "sign-return-address-key"="a_key" }
attributes #2 = { noinline nounwind optnone "branch-target-enforcement"="false" "sign-return-address"="none" }
attributes #3 = { noinline nounwind optnone "branch-target-enforcement"="false" "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" }

;; Intentionally no module flags
