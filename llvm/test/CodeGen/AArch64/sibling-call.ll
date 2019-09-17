; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -aarch64-enable-ldst-opt=0 | FileCheck %s

declare void @callee_stack0()
declare void @callee_stack8([8 x i64], i64)
declare void @callee_stack16([8 x i64], i64, i64)

define void @caller_to0_from0() nounwind {
; CHECK-LABEL: caller_to0_from0:
; CHECK-NEXT: // %bb.
  tail call void @callee_stack0()
  ret void
; CHECK-NEXT: b callee_stack0
}

define void @caller_to0_from8([8 x i64], i64) nounwind{
; CHECK-LABEL: caller_to0_from8:
; CHECK-NEXT: // %bb.

  tail call void @callee_stack0()
  ret void
; CHECK-NEXT: b callee_stack0
}

define void @caller_to8_from0() {
; CHECK-LABEL: caller_to8_from0:

; Caller isn't going to clean up any extra stack we allocate, so it
; can't be a tail call.
  tail call void @callee_stack8([8 x i64] undef, i64 42)
  ret void
; CHECK: bl callee_stack8
}

define void @caller_to8_from8([8 x i64], i64 %a) {
; CHECK-LABEL: caller_to8_from8:
; CHECK-NOT: sub sp, sp,

; This should reuse our stack area for the 42
  tail call void @callee_stack8([8 x i64] undef, i64 42)
  ret void
; CHECK: str {{x[0-9]+}}, [sp]
; CHECK-NEXT: b callee_stack8
}

define void @caller_to16_from8([8 x i64], i64 %a) {
; CHECK-LABEL: caller_to16_from8:

; Shouldn't be a tail call: we can't use SP+8 because our caller might
; have something there. This may sound obvious but implementation does
; some funky aligning.
  tail call void @callee_stack16([8 x i64] undef, i64 undef, i64 undef)
; CHECK: bl callee_stack16
  ret void
}

define void @caller_to8_from24([8 x i64], i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: caller_to8_from24:
; CHECK-NOT: sub sp, sp

; Reuse our area, putting "42" at incoming sp
  tail call void @callee_stack8([8 x i64] undef, i64 42)
  ret void
; CHECK: str {{x[0-9]+}}, [sp]
; CHECK-NEXT: b callee_stack8
}

define void @caller_to16_from16([8 x i64], i64 %a, i64 %b) {
; CHECK-LABEL: caller_to16_from16:
; CHECK-NOT: sub sp, sp,

; Here we want to make sure that both loads happen before the stores:
; otherwise either %a or %b will be wrongly clobbered.
  tail call void @callee_stack16([8 x i64] undef, i64 %b, i64 %a)
  ret void

; CHECK: ldr [[VAL0:x[0-9]+]],
; CHECK: ldr [[VAL1:x[0-9]+]],
; CHECK: str [[VAL0]],
; CHECK: str [[VAL1]],

; CHECK-NOT: add sp, sp,
; CHECK: b callee_stack16
}

@func = global void(i32)* null

define void @indirect_tail() {
; CHECK-LABEL: indirect_tail:
; CHECK-NOT: sub sp, sp

  %fptr = load void(i32)*, void(i32)** @func
  tail call void %fptr(i32 42)
  ret void
; CHECK: ldr [[FPTR:x[1-9]+]], [{{x[0-9]+}}, {{#?}}:lo12:func]
; CHECK: mov w0, #{{42|0x2a}}
; CHECK: br [[FPTR]]
}
