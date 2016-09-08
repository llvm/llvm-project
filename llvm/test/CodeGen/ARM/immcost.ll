; RUN: llc %s -o - -O1 -debug-only=consthoist 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv6m-apple-ios8.0.0"

declare void @g(i32)

; CHECK: Collect constant i32 -3 from   call void @g(i32 -3) with cost 2
define void @f(i1 %cond) {
entry:
  call void @g(i32 -3)
  br i1 %cond, label %true, label %ret

true:
  call void @g(i32 -3)
  br label %ret

ret:
  ret void
}

; CHECK: Function: h
; CHECK-NOT: Collect constant i32 -193 from
define void @h(i1 %cond, i32 %p, i32 %q) {
entry:
  %a = and i32 %p, 4294967103
  call void @g(i32 %a)
  br i1 %cond, label %true, label %ret

true:
  %b = and i32 %q, 4294967103
  call void @g(i32 %b)
  br label %ret

ret:
  ret void
}

