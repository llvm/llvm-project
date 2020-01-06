; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -tailcallopt | FileCheck %s --check-prefixes=SDAG,COMMON
; RUN: llc -global-isel -global-isel-abort=1 -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -tailcallopt | FileCheck %s --check-prefixes=GISEL,COMMON

declare fastcc void @callee_stack0()
declare fastcc void @callee_stack8([8 x i64], i64)
declare fastcc void @callee_stack16([8 x i64], i64, i64)
declare extern_weak fastcc void @callee_weak()

define fastcc void @caller_to0_from0() nounwind {
; COMMON-LABEL: caller_to0_from0:
; COMMON-NEXT: // %bb.

  tail call fastcc void @callee_stack0()
  ret void

; COMMON-NEXT: b callee_stack0
}

define fastcc void @caller_to0_from8([8 x i64], i64) {
; COMMON-LABEL: caller_to0_from8:

  tail call fastcc void @callee_stack0()
  ret void

; COMMON: add sp, sp, #16
; COMMON-NEXT: b callee_stack0
}

define fastcc void @caller_to8_from0() {
; COMMON-LABEL: caller_to8_from0:
; COMMON: sub sp, sp, #32

; Key point is that the "42" should go #16 below incoming stack
; pointer (we didn't have arg space to reuse).
  tail call fastcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void

; COMMON: str {{x[0-9]+}}, [sp, #16]!
; COMMON-NEXT: b callee_stack8
}

define fastcc void @caller_to8_from8([8 x i64], i64 %a) {
; COMMON-LABEL: caller_to8_from8:
; COMMON: sub sp, sp, #16

; Key point is that the "%a" should go where at SP on entry.
  tail call fastcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void

; COMMON: str {{x[0-9]+}}, [sp, #16]!
; COMMON-NEXT: b callee_stack8
}

define fastcc void @caller_to16_from8([8 x i64], i64 %a) {
; COMMON-LABEL: caller_to16_from8:
; COMMON: sub sp, sp, #16

; Important point is that the call reuses the "dead" argument space
; above %a on the stack. If it tries to go below incoming-SP then the
; callee will not deallocate the space, even in fastcc.
  tail call fastcc void @callee_stack16([8 x i64] undef, i64 42, i64 2)

; COMMON: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]!
; COMMON-NEXT: b callee_stack16
  ret void
}


define fastcc void @caller_to8_from24([8 x i64], i64 %a, i64 %b, i64 %c) {
; COMMON-LABEL: caller_to8_from24:
; COMMON: sub sp, sp, #16

; Key point is that the "%a" should go where at #16 above SP on entry.
  tail call fastcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void

; COMMON: str {{x[0-9]+}}, [sp, #32]!
; COMMON-NEXT: b callee_stack8
}


define fastcc void @caller_to16_from16([8 x i64], i64 %a, i64 %b) {
; COMMON-LABEL: caller_to16_from16:
; COMMON: sub sp, sp, #16

; Here we want to make sure that both loads happen before the stores:
; otherwise either %a or %b will be wrongly clobbered.
  tail call fastcc void @callee_stack16([8 x i64] undef, i64 %b, i64 %a)
  ret void

; COMMON: ldp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]
; COMMON: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]!
; COMMON-NEXT: b callee_stack16
}

define fastcc void @disable_tail_calls() nounwind "disable-tail-calls"="true" {
; COMMON-LABEL: disable_tail_calls:
; COMMON-NEXT: // %bb.

  tail call fastcc void @callee_stack0()
  ret void

; COMMON: bl callee_stack0
; COMMON: ret
}

; Weakly-referenced extern functions cannot be tail-called, as AAELF does
; not define the behaviour of branch instructions to undefined weak symbols.
define fastcc void @caller_weak() {
; COMMON-LABEL: caller_weak:
; COMMON: bl callee_weak
  tail call void @callee_weak()
  ret void
}

declare { [2 x float] } @get_vec2()

define { [3 x float] } @test_add_elem() {
; SDAG-LABEL: test_add_elem:
; SDAG: bl get_vec2
; SDAG: fmov s2, #1.0
; SDAG: ret
; GISEL-LABEL: test_add_elem:
; GISEL: fmov	s8, #1.00000000
; GISEL: bl get_vec2
; GISEL: ldr	x30, [sp, #8]
; GISEL: mov	v2.16b, v8.16b
; GISEL: ldr	d8, [sp], #16
; GISEL: ret

  %call = tail call { [2 x float] } @get_vec2()
  %arr = extractvalue { [2 x float] } %call, 0
  %arr.0 = extractvalue [2 x float] %arr, 0
  %arr.1 = extractvalue [2 x float] %arr, 1

  %res.0 = insertvalue { [3 x float] } undef, float %arr.0, 0, 0
  %res.01 = insertvalue { [3 x float] } %res.0, float %arr.1, 0, 1
  %res.012 = insertvalue { [3 x float] } %res.01, float 1.000000e+00, 0, 2
  ret { [3 x float] } %res.012
}

declare double @get_double()
define { double, [2 x double] } @test_mismatched_insert() {
; COMMON-LABEL: test_mismatched_insert:
; COMMON: bl get_double
; COMMON: bl get_double
; COMMON: bl get_double
; COMMON: ret

  %val0 = call double @get_double()
  %val1 = call double @get_double()
  %val2 = tail call double @get_double()

  %res.0 = insertvalue { double, [2 x double] } undef, double %val0, 0
  %res.01 = insertvalue { double, [2 x double] } %res.0, double %val1, 1, 0
  %res.012 = insertvalue { double, [2 x double] } %res.01, double %val2, 1, 1

  ret { double, [2 x double] } %res.012
}
