; RUN: llc -verify-machineinstrs < %s -mtriple=thumbv7k-apple-watchos | FileCheck %s

declare tailcc void @callee_stack0()
declare tailcc void @callee_stack4([4 x i32], i32)
declare tailcc void @callee_stack20([4 x i32], [5 x i32])
declare extern_weak tailcc void @callee_weak()

define tailcc void @caller_to0_from0() nounwind {
; CHECK-LABEL: _caller_to0_from0:

  tail call tailcc void @callee_stack0()
  ret void
; CHECK-NOT: add
; CHECK-NOT: sub
; CHECK: b.w _callee_stack0
}

define tailcc void @caller_to0_from4([4 x i32], i32) {
; CHECK-LABEL: _caller_to0_from4:

  tail call tailcc void @callee_stack0()
  ret void

; CHECK: add sp, #16
; CHECK-NEXT: b.w _callee_stack0
}

define tailcc void @caller_to4_from0() {
; Key point is that the "42" should go #16 below incoming stack
; pointer (we didn't have arg space to reuse).
  tail call tailcc void @callee_stack4([4 x i32] undef, i32 42)
  ret void

; CHECK-LABEL: _caller_to4_from0:
; CHECK: sub sp, #16
; CHECK: movs [[TMP:r[0-9]+]], #42
; CHECK: str [[TMP]], [sp]
; CHECK-NOT: add sp
; CHECK: b.w _callee_stack4

}

define tailcc void @caller_to4_from4([4 x i32], i32 %a) {
; CHECK-LABEL: _caller_to4_from4:
; CHECK-NOT: sub sp
; Key point is that the "%a" should go where at SP on entry.
  tail call tailcc void @callee_stack4([4 x i32] undef, i32 42)
  ret void

; CHECK: str {{r[0-9]+}}, [sp]
; CHECK-NOT: add sp
; CHECK-NEXT: b.w _callee_stack4
}

define tailcc void @caller_to20_from4([4 x i32], i32 %a) {
; CHECK-LABEL: _caller_to20_from4:
; CHECK: sub sp, #16

; Important point is that the call reuses the "dead" argument space
; above %a on the stack. If it tries to go below incoming-SP then the
; _callee will not deallocate the space, even in tailcc.
  tail call tailcc void @callee_stack20([4 x i32] undef, [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5])

; CHECK: str {{.*}}, [sp]
; CHECK: str {{.*}}, [sp, #4]
; CHECK: str {{.*}}, [sp, #8]
; CHECK: str {{.*}}, [sp, #12]
; CHECK: str {{.*}}, [sp, #16]
; CHECK-NOT: add sp
; CHECK-NOT: sub sp
; CHECK: b.w _callee_stack20
  ret void
}


define tailcc void @caller_to4_from24([4 x i32], i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: _caller_to4_from24:


; Key point is that the "%a" should go where at #16 above SP on entry.
  tail call tailcc void @callee_stack4([4 x i32] undef, i32 42)
  ret void

; CHECK: str {{.*}}, [sp, #16]
; CHECK: add sp, #16
; CHECK-NEXT: b.w _callee_stack4
}


define tailcc void @caller_to20_from20([4 x i32], [5 x i32] %a) {
; CHECK-LABEL: _caller_to20_from20:
; CHECK-NOT: add sp,
; CHECK-NOT: sub sp,

; Here we want to make sure that both loads happen before the stores:
; otherwise either %a or %b.w will be wrongly clobbered.
  tail call tailcc void @callee_stack20([4 x i32] undef, [5 x i32] %a)
  ret void

  ; If these ever get interleaved make sure aliasing slots don't clobber each
  ; other.
; CHECK: ldrd {{.*}}, {{.*}}, [sp, #12]
; CHECK: ldm.w sp,
; CHECK: stm.w
; CHECK: strd
; CHECK-NEXT: b.w _callee_stack20
}

define tailcc void @disable_tail_calls() nounwind "disable-tail-calls"="true" {
; CHECK-LABEL: disable_tail_calls:

  tail call tailcc void @callee_stack0()
  ret void

; CHECK: bl _callee_stack0
; CHECK: ret
}

define tailcc void @normal_ret_with_stack([4 x i32], i32 %a) {
; CHECK: _normal_ret_with_stack:
; CHECK: add sp, #16
; CHECK: bx lr
  ret void
}

declare { [2 x float] } @get_vec2()

define void @fromC_totail() {
; COMMON-LABEL: fromC_totail:
; COMMON: puch {r4, lr}
; COMMON: sub sp, #8

; COMMON-NOT: sub sp,
; COMMON: movs [[TMP:r[0-9]+]], #42
; COMMON: str [[TMP]], [sp]
; COMMON: bl _callee_stack4
  ; We must reset the stack to where it was before the call by undoing its extra stack pop.
; COMMON: sub sp, #16
; COMMON: str [[TMP]], [sp]
; COMMON: bl callee_stack4
; COMMON: sub sp, #16

  call tailcc void @callee_stack4([4 x i32] undef, i32 42)
  call tailcc void @callee_stack4([4 x i32] undef, i32 42)
  ret void
}

define void @fromC_totail_noreservedframe(i32 %len) {
; COMMON-LABEL: fromC_totail_noreservedframe:
; COMMON: sub.w sp, sp, r{{.*}}

; COMMON: movs [[TMP:r[0-9]+]], #42
  ; Note stack is subtracted here to allocate space for arg
; COMMON: sub.w sp, #16
; COMMON: str [[TMP]], [sp]
; COMMON: bl _callee_stack4
  ; And here.
; COMMON: sub sp, #16
; COMMON: str [[TMP]], [sp]
; COMMON: bl _callee_stack4
  ; But not restored here because callee_stack8 did that for us.
; COMMON-NOT: sub sp,

  ; Variable sized allocation prevents reserving frame at start of function so each call must allocate any stack space it needs.
  %var = alloca i32, i32 %len

  call tailcc void @callee_stack4([4 x i32] undef, i32 42)
  call tailcc void @callee_stack4([4 x i32] undef, i32 42)
  ret void
}

declare void @Ccallee_stack4([4 x i32], i32)

define tailcc void @fromtail_toC() {
; COMMON-LABEL: fromtail_toC:
; COMMON: push {r4, lr}
; COMMON: sub sp, #8

; COMMON-NOT: sub sp,
; COMMON: movs [[TMP:r[0-9]+]], #42
; COMMON: str [[TMP]], [sp]
; COMMON: bl _Ccallee_stack4
  ; C callees will return with the stack exactly where we left it, so we mustn't try to fix anything.
; COMMON-NOT: add sp,
; COMMON-NOT: sub sp,
; COMMON: str [[TMP]], [sp]{{$}}
; COMMON: bl _Ccallee_stack4
; COMMON-NOT: sub sp,

  call void @Ccallee_stack4([4 x i32] undef, i32 42)
  call void @Ccallee_stack4([4 x i32] undef, i32 42)
  ret void
}
