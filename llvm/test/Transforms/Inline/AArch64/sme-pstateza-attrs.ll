; RUN: opt -mtriple=aarch64-unknown-linux-gnu -mattr=+sme -S -passes=inline < %s | FileCheck %s

declare void @inlined_body()

;
; Define some functions that will be called by the functions below.
; These just call a '...body()' function. If we see the call to one of
; these functions being replaced by '...body()', then we know it has been
; inlined.
;

define void @nonza_callee() {
entry:
  call void @inlined_body()
  ret void
}

define void @shared_za_callee() "aarch64_pstate_za_shared" {
entry:
  call void @inlined_body()
  ret void
}

define void @new_za_callee() "aarch64_pstate_za_new" {
  call void @inlined_body()
  ret void
}

;
; Now test that inlining only happens when no lazy-save is needed.
; Test for a number of combinations, where:
; N   Not using ZA.
; S   Shared ZA interface
; Z   New ZA interface

; [x] N -> N
; [ ] N -> S (This combination is invalid)
; [ ] N -> Z
define void @nonza_caller_nonza_callee_inline() {
; CHECK-LABEL: @nonza_caller_nonza_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @nonza_callee()
  ret void
}

; [ ] N -> N
; [ ] N -> S (This combination is invalid)
; [x] N -> Z
define void @nonza_caller_new_za_callee_dont_inline() {
; CHECK-LABEL: @nonza_caller_new_za_callee_dont_inline(
; CHECK: call void @new_za_callee()
entry:
  call void @new_za_callee()
  ret void
}

; [x] Z -> N
; [ ] Z -> S
; [ ] Z -> Z
define void @new_za_caller_nonza_callee_dont_inline() "aarch64_pstate_za_new" {
; CHECK-LABEL: @new_za_caller_nonza_callee_dont_inline(
; CHECK: call void @nonza_callee()
entry:
  call void @nonza_callee()
  ret void
}

; [ ] Z -> N
; [x] Z -> S
; [ ] Z -> Z
define void @new_za_caller_shared_za_callee_inline() "aarch64_pstate_za_new" {
; CHECK-LABEL: @new_za_caller_shared_za_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @shared_za_callee()
  ret void
}

; [ ] Z -> N
; [ ] Z -> S
; [x] Z -> Z
define void @new_za_caller_new_za_callee_dont_inline() "aarch64_pstate_za_new" {
; CHECK-LABEL: @new_za_caller_new_za_callee_dont_inline(
; CHECK: call void @new_za_callee()
entry:
  call void @new_za_callee()
  ret void
}

; [x] Z -> N
; [ ] Z -> S
; [ ] Z -> Z
define void @shared_za_caller_nonza_callee_dont_inline() "aarch64_pstate_za_shared" {
; CHECK-LABEL: @shared_za_caller_nonza_callee_dont_inline(
; CHECK: call void @nonza_callee()
entry:
  call void @nonza_callee()
  ret void
}

; [ ] S -> N
; [x] S -> Z
; [ ] S -> S
define void @shared_za_caller_new_za_callee_dont_inline() "aarch64_pstate_za_shared" {
; CHECK-LABEL: @shared_za_caller_new_za_callee_dont_inline(
; CHECK: call void @new_za_callee()
entry:
  call void @new_za_callee()
  ret void
}

; [ ] S -> N
; [ ] S -> Z
; [x] S -> S
define void @shared_za_caller_shared_za_callee_inline() "aarch64_pstate_za_shared" {
; CHECK-LABEL: @shared_za_caller_shared_za_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @shared_za_callee()
  ret void
}
