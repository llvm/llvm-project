; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s --check-prefixes=SDAG,COMMON
; RUN: llc -global-isel -global-isel-abort=1 -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu  | FileCheck %s --check-prefixes=GISEL,COMMON

declare tailcc void @callee_stack0()
declare tailcc void @callee_stack8([8 x i64], i64)
declare tailcc void @callee_stack16([8 x i64], i64, i64)
declare extern_weak tailcc void @callee_weak()

define tailcc void @caller_to0_from0() nounwind {
; COMMON-LABEL: caller_to0_from0:
; COMMON-NEXT: // %bb.

  tail call tailcc void @callee_stack0()
  ret void

; COMMON-NEXT: b callee_stack0
}

define tailcc void @caller_to0_from8([8 x i64], i64) {
; COMMON-LABEL: caller_to0_from8:

  tail call tailcc void @callee_stack0()
  ret void

; COMMON: add sp, sp, #16
; COMMON-NEXT: b callee_stack0
}

define tailcc void @caller_to8_from0() "frame-pointer"="all"{
; COMMON-LABEL: caller_to8_from0:

; Key point is that the "42" should go #16 below incoming stack
; pointer (we didn't have arg space to reuse).
  tail call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void

; COMMON: str {{x[0-9]+}}, [x29, #16]
; COMMON: ldp x29, x30, [sp], #16
  ; If there is a sub here then the 42 will be briefly exposed to corruption
  ; from an interrupt if the kernel does not honour a red-zone, and a larger
  ; call could well overflow the red zone even if it is present.
; COMMON-NOT: sub sp,
; COMMON-NEXT: b callee_stack8
}

define tailcc void @caller_to8_from8([8 x i64], i64 %a) {
; COMMON-LABEL: caller_to8_from8:
; COMMON-NOT: sub sp,

; Key point is that the "%a" should go where at SP on entry.
  tail call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void

; COMMON: str {{x[0-9]+}}, [sp]
; COMMON-NEXT: b callee_stack8
}

define tailcc void @caller_to16_from8([8 x i64], i64 %a) {
; COMMON-LABEL: caller_to16_from8:
; COMMON-NOT: sub sp,

; Important point is that the call reuses the "dead" argument space
; above %a on the stack. If it tries to go below incoming-SP then the
; callee will not deallocate the space, even in tailcc.
  tail call tailcc void @callee_stack16([8 x i64] undef, i64 42, i64 2)

; COMMON: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp]
; COMMON-NEXT: b callee_stack16
  ret void
}


define tailcc void @caller_to8_from24([8 x i64], i64 %a, i64 %b, i64 %c) {
; COMMON-LABEL: caller_to8_from24:
; COMMON-NOT: sub sp,

; Key point is that the "%a" should go where at #16 above SP on entry.
  tail call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void

; COMMON: str {{x[0-9]+}}, [sp, #16]!
; COMMON-NEXT: b callee_stack8
}


define tailcc void @caller_to16_from16([8 x i64], i64 %a, i64 %b) {
; COMMON-LABEL: caller_to16_from16:
; COMMON-NOT: sub sp,

; Here we want to make sure that both loads happen before the stores:
; otherwise either %a or %b will be wrongly clobbered.
  tail call tailcc void @callee_stack16([8 x i64] undef, i64 %b, i64 %a)
  ret void

; COMMON: ldp {{x[0-9]+}}, {{x[0-9]+}}, [sp]
; COMMON: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp]
; COMMON-NEXT: b callee_stack16
}

define tailcc void @disable_tail_calls() nounwind "disable-tail-calls"="true" {
; COMMON-LABEL: disable_tail_calls:
; COMMON-NEXT: // %bb.

  tail call tailcc void @callee_stack0()
  ret void

; COMMON: bl callee_stack0
; COMMON: ret
}

; Weakly-referenced extern functions cannot be tail-called, as AAELF does
; not define the behaviour of branch instructions to undefined weak symbols.
define tailcc void @caller_weak() {
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
; GISEL: str	x30, [sp, #-16]!
; GISEL: bl get_vec2
; GISEL: fmov	s2, #1.0
; GISEL: ldr	x30, [sp], #16
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

define void @fromC_totail() {
; COMMON-LABEL: fromC_totail:
; COMMON: sub sp, sp, #32

; COMMON-NOT: sub sp,
; COMMON: mov w[[TMP:[0-9]+]], #42
; COMMON: str x[[TMP]], [sp]
; COMMON: bl callee_stack8
  ; We must reset the stack to where it was before the call by undoing its extra stack pop.
; COMMON: str x[[TMP]], [sp, #-16]!
; COMMON: bl callee_stack8
; COMMON: sub sp, sp, #16

  call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void
}

define void @fromC_totail_noreservedframe(i32 %len) {
; COMMON-LABEL: fromC_totail_noreservedframe:
; COMMON: stp x29, x30, [sp, #-32]!

; COMMON: mov w[[TMP:[0-9]+]], #42
  ; Note stack is subtracted here to allocate space for arg
; COMMON: str x[[TMP]], [sp, #-16]!
; COMMON: bl callee_stack8
  ; And here.
; COMMON: str x[[TMP]], [sp, #-16]!
; COMMON: bl callee_stack8
  ; But not restored here because callee_stack8 did that for us.
; COMMON-NOT: sub sp,

  ; Variable sized allocation prevents reserving frame at start of function so each call must allocate any stack space it needs.
  %var = alloca i32, i32 %len

  call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void
}

declare void @Ccallee_stack8([8 x i64], i64)

define tailcc void @fromtail_toC() {
; COMMON-LABEL: fromtail_toC:
; COMMON: sub sp, sp, #32

; COMMON-NOT: sub sp,
; COMMON: mov w[[TMP:[0-9]+]], #42
; COMMON: str x[[TMP]], [sp]
; COMMON: bl Ccallee_stack8
  ; C callees will return with the stack exactly where we left it, so we mustn't try to fix anything.
; COMMON-NOT: add sp,
; COMMON-NOT: sub sp,
; COMMON: str x[[TMP]], [sp]{{$}}
; COMMON: bl Ccallee_stack8
; COMMON-NOT: sub sp,


  call void @Ccallee_stack8([8 x i64] undef, i64 42)
  call void @Ccallee_stack8([8 x i64] undef, i64 42)
  ret void
}
