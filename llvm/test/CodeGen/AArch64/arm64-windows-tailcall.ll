; FIXME: Add tests for global-isel/fast-isel.

; RUN: llc < %s -mtriple=arm64-windows | FileCheck %s

%class.C = type { [1 x i32] }

define dso_local void @"?bar"(ptr inreg noalias sret(%class.C) %agg.result) {
entry:
; CHECK-LABEL: bar
; CHECK: mov x19, x0
; CHECK: bl "?foo"
; CHECK: mov x0, x19

  tail call void @"?foo"(ptr dereferenceable(4) %agg.result)
  ret void
}

declare dso_local void @"?foo"(ptr dereferenceable(4))


declare void @inreg_callee(ptr, ptr inreg sret(%class.C))

define void @inreg_caller_1(ptr %a, ptr inreg sret(%class.C) %b) {
; A different value is passed to the inreg parameter, so tail call is not possible.
; CHECK-LABEL: inreg_caller_1
; CHECK: mov x19, x1
; CHECK: bl inreg_callee
; CHECK: mov x0, x19

  tail call void @inreg_callee(ptr %b, ptr inreg sret(%class.C) %a)
  ret void
}

define void @inreg_caller_2(ptr %a, ptr inreg sret(%class.C) %b) {
; The inreg attribute and value line up between caller and callee, so it can
; be tail called.
; CHECK-LABEL: inreg_caller_2
; CHECK: b inreg_callee

  tail call void @inreg_callee(ptr %a, ptr inreg sret(%class.C) %b)
  ret void
}
