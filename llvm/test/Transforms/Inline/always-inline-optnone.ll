; RUN: opt < %s -passes=always-inline -S | FileCheck %s
; RUN: opt < %s -passes=inline -S | FileCheck %s

; optnone implies noinline, but alwaysinline takes precedence, so a callee with
; both attributes is inlined by both the always-inliner and the regular inliner.

define i32 @callee_alwaysinline_optnone(i32 %a) alwaysinline optnone {
  %r = add i32 %a, 1
  ret i32 %r
}

; optnone without alwaysinline is never inlined.
define i32 @callee_optnone(i32 %a) optnone {
  %r = add i32 %a, 2
  ret i32 %r
}

; CHECK-LABEL: define i32 @caller(
; CHECK-NOT: call i32 @callee_alwaysinline_optnone
; CHECK: call i32 @callee_optnone
; CHECK: ret
define i32 @caller(i32 %a) {
  %x = call i32 @callee_alwaysinline_optnone(i32 %a)
  %y = call i32 @callee_optnone(i32 %x)
  ret i32 %y
}

; A call-site noinline still overrides alwaysinline.
; CHECK-LABEL: define i32 @caller_noinline_callsite(
; CHECK: call i32 @callee_alwaysinline_optnone(i32 %a) #[[NOINLINE:[0-9]+]]
define i32 @caller_noinline_callsite(i32 %a) {
  %x = call i32 @callee_alwaysinline_optnone(i32 %a) noinline
  ret i32 %x
}

; CHECK: attributes #[[NOINLINE]] = { noinline }
