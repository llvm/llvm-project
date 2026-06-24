; RUN: opt < %s -passes=tailcallelim -verify-dom-info -S | FileCheck %s

; Reduced from gcc-c-torture/execute/pr17377.c
;
; A call must not be marked as tail if the callee uses @llvm.returnaddress
; or @llvm.frameaddress, because tail-call optimization destroys the
; caller's stack frame and changes what these intrinsics observe.

; The call from @y to @f must NOT be marked tail because @f uses
; @llvm.returnaddress.
define ptr @y(i32 %i) {
; CHECK-LABEL: define ptr @y
; CHECK: %call = call ptr @f(i32 %i)
; CHECK-NOT: tail call ptr @f
entry:
  %call = call ptr @f(i32 %i)
  ret ptr %call
}

define ptr @f(i32 %i) noinline {
entry:
  %addr = call ptr @llvm.returnaddress(i32 0)
  ret ptr %addr
}

; Same test but for @llvm.frameaddress — the call from @y2 to @g must
; NOT be marked tail.
define ptr @y2(i32 %i) {
; CHECK-LABEL: define ptr @y2
; CHECK: %call = call ptr @g(i32 %i)
; CHECK-NOT: tail call ptr @g
entry:
  %call = call ptr @g(i32 %i)
  ret ptr %call
}

define ptr @g(i32 %i) noinline {
entry:
  %addr = call ptr @llvm.frameaddress(i32 0)
  ret ptr %addr
}

; Negative test: a callee that does NOT use returnaddress/frameaddress
; SHOULD still be marked tail.
define void @caller_normal() {
; CHECK-LABEL: define void @caller_normal
; CHECK: tail call void @normal_callee()
entry:
  call void @normal_callee()
  ret void
}

declare void @normal_callee()
declare ptr @llvm.returnaddress(i32 immarg)
declare ptr @llvm.frameaddress(i32 immarg)
