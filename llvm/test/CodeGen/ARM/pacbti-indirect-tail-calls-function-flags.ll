; RUN: llc -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+pacbti< %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-m.main-unknown"

; When PACBTI is enabled, indirect tail-calls must not use R12 that is used
; to store authentication code.

define void @pacbti_disabled(ptr %p) "sign-return-address"="none" {
entry:
  tail call void %p()
; CHECK: bx {{r0|r1|r2|r3|r12}}
  ret void
}

define void @pacbti_enabled(ptr %p) "sign-return-address"="all" {
entry:
  tail call void %p()
; CHECK: bx {{r0|r1|r2|r3}}
  ret void
}

define void @pacbti_disabled_force_r12(ptr %p) "sign-return-address"="none" {
entry:
  %p_r12 = tail call ptr asm "", "={r12},{r12},~{lr}"(ptr %p)
  tail call void %p_r12()
; CHECK: bx r12
  ret void
}

define void @pacbti_enabled_force_r12(ptr %p) "sign-return-address"="all" {
entry:
  %p_r12 = tail call ptr asm "", "={r12},{r12},~{lr}"(ptr %p)
  tail call void %p_r12()
; CHECK: bx {{r0|r1|r2|r3}}
  ret void
}
