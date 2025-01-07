; RUN: llc -mtriple aarch64 -mattr=+bti < %s | FileCheck %s
; RUN: llc -mtriple aarch64 -global-isel -mattr=+bti < %s | FileCheck %s

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

define void @bti_enabled(ptr %p) "branch-target-enforcement" {
entry:
  tail call void %p()
; CHECK: br {{x16|x17}}
  ret void
}
define void @bti_enabled_force_x10(ptr %p) "branch-target-enforcement" {
entry:
  %p_x10 = tail call ptr asm "", "={x10},{x10},~{lr}"(ptr %p)
  tail call void %p_x10()
; CHECK: br {{x16|x17}}
  ret void
}

; sign-return-address places no further restrictions on the tail-call register.

define void @bti_enabled_pac_enabled(ptr %p) "branch-target-enforcement" "sign-return-address"="all" {
entry:
  tail call void %p()
; CHECK: br {{x16|x17}}
  ret void
}
define void @bti_enabled_pac_enabled_force_x10(ptr %p) "branch-target-enforcement" "sign-return-address"="all" {
entry:
  %p_x10 = tail call ptr asm "", "={x10},{x10},~{lr}"(ptr %p)
  tail call void %p_x10()
; CHECK: br {{x16|x17}}
  ret void
}

; PAuthLR needs to use x16 to hold the address of the signing instruction. That
; can't be changed because the hint instruction only uses that register, so the
; only choice for the tail-call function pointer is x17.

define void @bti_enabled_pac_pc_enabled(ptr %p) "branch-target-enforcement" "sign-return-address"="all" "branch-protection-pauth-lr" {
entry:
  tail call void %p()
; CHECK: br x17
  ret void
}
define void @bti_enabled_pac_pc_enabled_force_x16(ptr %p) "branch-target-enforcement" "sign-return-address"="all" "branch-protection-pauth-lr" {
entry:
  %p_x16 = tail call ptr asm "", "={x16},{x16},~{lr}"(ptr %p)
  tail call void %p_x16()
; CHECK: br x17
  ret void
}

; PAuthLR by itself prevents x16 from being used, but any other
; non-callee-saved register can be used.

define void @pac_pc_enabled(ptr %p) "sign-return-address"="all" "branch-protection-pauth-lr" {
entry:
  tail call void %p()
; CHECK: br {{(x[0-9]|x1[0-578])$}}
  ret void
}
define void @pac_pc_enabled_force_x16(ptr %p) "sign-return-address"="all" "branch-protection-pauth-lr" {
entry:
  %p_x16 = tail call ptr asm "", "={x16},{x16},~{lr}"(ptr %p)
  tail call void %p_x16()
; CHECK: br {{(x[0-9]|x1[0-578])$}}
  ret void
}
define void @pac_pc_enabled_force_x17(ptr %p) "sign-return-address"="all" "branch-protection-pauth-lr" {
entry:
  %p_x17 = tail call ptr asm "", "={x17},{x17},~{lr}"(ptr %p)
  tail call void %p_x17()
; CHECK: br x17
  ret void
}
