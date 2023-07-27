; RUN: llc -mtriple aarch64 -mattr=+bti < %s | FileCheck %s
; RUN: llc -mtriple aarch64 -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* -mattr=+bti %s -verify-machineinstrs -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,FALLBACK

; FALLBACK: remark: <unknown>:0:0: unable to translate instruction: call: '  tail call void %p()' (in function: bti_enabled)

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; When BTI is enabled, all indirect tail-calls must use x16 or x17 (the intra
; procedure call scratch registers) to hold the address, as these instructions
; are allowed to target the "BTI c" instruction at the start of the target
; function. The alternative to this would be to start functions with "BTI jc",
; which increases the number of potential ways they could be called, and
; weakens the security protections.

define void @bti_disabled(ptr %p) {
entry:
  tail call void %p()
; CHECK: br x0
  ret void
}

define void @bti_enabled(ptr %p) "branch-target-enforcement"="true" {
entry:
  tail call void %p()
; CHECK: br {{x16|x17}}
  ret void
}
