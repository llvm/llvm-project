; RUN: opt < %s -mtriple=aarch64-unknown-linux-gnu -mattr=+sme -S -inline | FileCheck %s

declare void @inlined_body() "aarch64_pstate_sm_compatible";

;
; Define some functions that will be called by the functions below.
; These just call a '...body()' function. If we see the call to one of
; these functions being replaced by '...body()', then we know it has been
; inlined.
;

define void @normal_callee() {
entry:
  call void @inlined_body()
  ret void
}

define void @streaming_callee() "aarch64_pstate_sm_enabled" {
entry:
  call void @inlined_body()
  ret void
}

define void @locally_streaming_callee() "aarch64_pstate_sm_body" {
entry:
  call void @inlined_body()
  ret void
}

define void @streaming_compatible_callee() "aarch64_pstate_sm_compatible" {
entry:
  call void @inlined_body()
  ret void
}

define void @streaming_compatible_locally_streaming_callee() "aarch64_pstate_sm_compatible" "aarch64_pstate_sm_body" {
entry:
  call void @inlined_body()
  ret void
}

;
; Now test that inlining only happens when their streaming modes match.
; Test for a number of combinations, where:
; N       Normal-interface (PSTATE.SM=0 on entry/exit)
; S       Streaming-interface (PSTATE.SM=1 on entry/exit)
; SC      Streaming-compatible interface
;         (PSTATE.SM=0 or 1, unchanged on exit)
; N + B   Normal-interface, streaming body
;         (PSTATE.SM=0 on entry/exit, but 1 within the body of the function)
; SC + B  Streaming-compatible-interface, streaming body
;         (PSTATE.SM=0 or 1 on entry, unchanged on exit,
;          but guaranteed to be 1 within the body of the function)

; [x] N  -> N
; [ ] N  -> S
; [ ] N  -> SC
; [ ] N  -> N + B
; [ ] N  -> SC + B
define void @normal_caller_normal_callee_inline() {
; CHECK-LABEL: @normal_caller_normal_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @normal_callee()
  ret void
}

; [ ] N  -> N
; [x] N  -> S
; [ ] N  -> SC
; [ ] N  -> N + B
; [ ] N  -> SC + B
define void @normal_caller_streaming_callee_dont_inline() {
; CHECK-LABEL: @normal_caller_streaming_callee_dont_inline(
; CHECK: call void @streaming_callee()
entry:
  call void @streaming_callee()
  ret void
}

; [ ] N  -> N
; [ ] N  -> S
; [x] N  -> SC
; [ ] N  -> N + B
; [ ] N  -> SC + B
define void @normal_caller_streaming_compatible_callee_inline() {
; CHECK-LABEL: @normal_caller_streaming_compatible_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_compatible_callee()
  ret void
}

; [ ] N  -> N
; [ ] N  -> S
; [ ] N  -> SC
; [x] N  -> N + B
; [ ] N  -> SC + B
define void @normal_caller_locally_streaming_callee_dont_inline() {
; CHECK-LABEL: @normal_caller_locally_streaming_callee_dont_inline(
; CHECK: call void @locally_streaming_callee()
entry:
  call void @locally_streaming_callee()
  ret void
}

; [ ] N  -> N
; [ ] N  -> S
; [ ] N  -> SC
; [ ] N  -> N + B
; [x] N  -> SC + B
define void @normal_caller_streaming_compatible_locally_streaming_callee_dont_inline() {
; CHECK-LABEL: @normal_caller_streaming_compatible_locally_streaming_callee_dont_inline(
; CHECK: call void @streaming_compatible_locally_streaming_callee()
entry:
  call void @streaming_compatible_locally_streaming_callee()
  ret void
}

; [x] S  -> N
; [ ] S  -> S
; [ ] S  -> SC
; [ ] S  -> N + B
; [ ] S  -> SC + B
define void @streaming_caller_normal_callee_dont_inline() "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: @streaming_caller_normal_callee_dont_inline(
; CHECK: call void @normal_callee()
entry:
  call void @normal_callee()
  ret void
}

; [ ] S  -> N
; [x] S  -> S
; [ ] S  -> SC
; [ ] S  -> N + B
; [ ] S  -> SC + B
define void @streaming_caller_streaming_callee_inline() "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: @streaming_caller_streaming_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_callee()
  ret void
}

; [ ] S  -> N
; [ ] S  -> S
; [x] S  -> SC
; [ ] S  -> N + B
; [ ] S  -> SC + B
define void @streaming_caller_streaming_compatible_callee_inline() "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: @streaming_caller_streaming_compatible_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_compatible_callee()
  ret void
}

; [ ] S  -> N
; [ ] S  -> S
; [ ] S  -> SC
; [x] S  -> N + B
; [ ] S  -> SC + B
define void @streaming_caller_locally_streaming_callee_inline() "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: @streaming_caller_locally_streaming_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @locally_streaming_callee()
  ret void
}

; [ ] S  -> N
; [ ] S  -> S
; [ ] S  -> SC
; [ ] S  -> N + B
; [x] S  -> SC + B
define void @streaming_caller_streaming_compatible_locally_streaming_callee_inline() "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: @streaming_caller_streaming_compatible_locally_streaming_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_compatible_locally_streaming_callee()
  ret void
}

; [x] N + B -> N
; [ ] N + B -> S
; [ ] N + B -> SC
; [ ] N + B -> N + B
; [ ] N + B -> SC + B
define void @locally_streaming_caller_normal_callee_dont_inline() "aarch64_pstate_sm_body" {
; CHECK-LABEL: @locally_streaming_caller_normal_callee_dont_inline(
; CHECK: call void @normal_callee()
entry:
  call void @normal_callee()
  ret void
}

; [ ] N + B -> N
; [x] N + B -> S
; [ ] N + B -> SC
; [ ] N + B -> N + B
; [ ] N + B -> SC + B
define void @locally_streaming_caller_streaming_callee_inline() "aarch64_pstate_sm_body" {
; CHECK-LABEL: @locally_streaming_caller_streaming_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_callee()
  ret void
}

; [ ] N + B -> N
; [ ] N + B -> S
; [x] N + B -> SC
; [ ] N + B -> N + B
; [ ] N + B -> SC + B
define void @locally_streaming_caller_streaming_compatible_callee_inline() "aarch64_pstate_sm_body" {
; CHECK-LABEL: @locally_streaming_caller_streaming_compatible_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_compatible_callee()
  ret void
}

; [ ] N + B -> N
; [ ] N + B -> S
; [ ] N + B -> SC
; [x] N + B -> N + B
; [ ] N + B -> SC + B
define void @locally_streaming_caller_locally_streaming_callee_inline() "aarch64_pstate_sm_body" {
; CHECK-LABEL: @locally_streaming_caller_locally_streaming_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @locally_streaming_callee()
  ret void
}

; [ ] N + B -> N
; [ ] N + B -> S
; [ ] N + B -> SC
; [ ] N + B -> N + B
; [x] N + B -> SC + B
define void @locally_streaming_caller_streaming_compatible_locally_streaming_callee_inline() "aarch64_pstate_sm_body" {
; CHECK-LABEL: @locally_streaming_caller_streaming_compatible_locally_streaming_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_compatible_locally_streaming_callee()
  ret void
}

; [x] SC -> N
; [ ] SC -> S
; [ ] SC -> SC
; [ ] SC -> N + B
; [ ] SC -> SC + B
define void @streaming_compatible_caller_normal_callee_dont_inline() "aarch64_pstate_sm_compatible" {
; CHECK-LABEL: @streaming_compatible_caller_normal_callee_dont_inline(
; CHECK: call void @normal_callee()
entry:
  call void @normal_callee()
  ret void
}

; [ ] SC -> N
; [x] SC -> S
; [ ] SC -> SC
; [ ] SC -> N + B
; [ ] SC -> SC + B
define void @streaming_compatible_caller_streaming_callee_dont_inline() "aarch64_pstate_sm_compatible" {
; CHECK-LABEL: @streaming_compatible_caller_streaming_callee_dont_inline(
; CHECK: call void @streaming_callee()
entry:
  call void @streaming_callee()
  ret void
}

; [ ] SC -> N
; [ ] SC -> S
; [x] SC -> SC
; [ ] SC -> N + B
; [ ] SC -> SC + B
define void @streaming_compatible_caller_streaming_compatible_callee_inline() "aarch64_pstate_sm_compatible" {
; CHECK-LABEL: @streaming_compatible_caller_streaming_compatible_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_compatible_callee()
  ret void
}

; [ ] SC -> N
; [ ] SC -> S
; [ ] SC -> SC
; [x] SC -> N + B
; [ ] SC -> SC + B
define void @streaming_compatible_caller_locally_streaming_callee_dont_inline() "aarch64_pstate_sm_compatible" {
; CHECK-LABEL: @streaming_compatible_caller_locally_streaming_callee_dont_inline(
; CHECK: call void @locally_streaming_callee()
entry:
  call void @locally_streaming_callee()
  ret void
}

; [ ] SC -> N
; [ ] SC -> S
; [ ] SC -> SC
; [ ] SC -> N + B
; [x] SC -> SC + B
define void @streaming_compatible_caller_streaming_compatible_locally_streaming_callee_dont_inline() "aarch64_pstate_sm_compatible" {
; CHECK-LABEL: @streaming_compatible_caller_streaming_compatible_locally_streaming_callee_dont_inline(
; CHECK: call void @streaming_compatible_locally_streaming_callee()
entry:
  call void @streaming_compatible_locally_streaming_callee()
  ret void
}
; [x] SC + B -> N
; [ ] SC + B -> S
; [ ] SC + B -> SC
; [ ] SC + B -> N + B
; [ ] SC + B -> SC + B
define void @streaming_compatible_locally_streaming_caller_normal_callee_dont_inline() "aarch64_pstate_sm_compatible" "aarch64_pstate_sm_body" {
; CHECK-LABEL: @streaming_compatible_locally_streaming_caller_normal_callee_dont_inline(
; CHECK: call void @normal_callee()
entry:
  call void @normal_callee()
  ret void
}

; [ ] SC + B -> N
; [x] SC + B -> S
; [ ] SC + B -> SC
; [ ] SC + B -> N + B
; [ ] SC + B -> SC + B
define void @streaming_compatible_locally_streaming_caller_streaming_callee_inline() "aarch64_pstate_sm_compatible" "aarch64_pstate_sm_body" {
; CHECK-LABEL: @streaming_compatible_locally_streaming_caller_streaming_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_callee()
  ret void
}

; [ ] SC + B -> N
; [ ] SC + B -> S
; [x] SC + B -> SC
; [ ] SC + B -> N + B
; [ ] SC + B -> SC + B
define void @streaming_compatible_locally_streaming_caller_streaming_compatible_callee_inline() "aarch64_pstate_sm_compatible" "aarch64_pstate_sm_body" {
; CHECK-LABEL: @streaming_compatible_locally_streaming_caller_streaming_compatible_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_compatible_callee()
  ret void
}

; [ ] SC + B -> N
; [ ] SC + B -> S
; [ ] SC + B -> SC
; [x] SC + B -> N + B
; [ ] SC + B -> SC + B
define void @streaming_compatible_locally_streaming_caller_locally_streaming_callee_inline() "aarch64_pstate_sm_compatible" "aarch64_pstate_sm_body" {
; CHECK-LABEL: @streaming_compatible_locally_streaming_caller_locally_streaming_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @locally_streaming_callee()
  ret void
}

; [ ] SC + B -> N
; [ ] SC + B -> S
; [ ] SC + B -> SC
; [ ] SC + B -> N + B
; [x] SC + B -> SC + B
define void @streaming_compatible_locally_streaming_caller_and_callee_inline() "aarch64_pstate_sm_compatible" "aarch64_pstate_sm_body" {
; CHECK-LABEL: @streaming_compatible_locally_streaming_caller_and_callee_inline(
; CHECK: call void @inlined_body()
entry:
  call void @streaming_compatible_locally_streaming_callee()
  ret void
}
