; This file contains the previous semantic of the branch-target-enforcement, sign-return-address.
; Used for test mixing a mixed link case and also verify the import too in llc.

; RUN: llc %s -o - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define i32 @old_bti() #0 {
entry:
  ret i32 2
}

; CHECK-LABEL: old_bti:
; CHECK:           hint    #34
; CHECK:           mov
; CHECK:           ret

define i32 @old_pac() #1 {
entry:
  ret i32 2
}

; CHECK-LABEL: old_pac:
; CHECK:           hint    #25
; CHECK:           hint    #29
; CHECK:           ret


define i32 @old_none() #2 {
entry:
  ret i32 3
}

; CHECK-LABEL: old_none:
; CHECK-NOT:           hint
; CHECK-NOT:           paci
; CHECK-NOT:           bti
; CHECK:           ret


attributes #0 = { noinline nounwind optnone "branch-target-enforcement"="true" }
attributes #1 = { noinline nounwind optnone "branch-target-enforcement"="false" "sign-return-address"="all" }
attributes #2 = { noinline nounwind optnone "branch-target-enforcement"="false" "sign-return-address"="none" }

; Intentionally no module flags
