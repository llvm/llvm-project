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

; CHECK: Function: test_icmp_neg
; CHECK-NOT: Collect constant
define void @test_icmp_neg(i1 %cond, i32 %arg, i32 %arg2) {
entry:
  %a = icmp ne i32 %arg, -5
  call void @g2(i1 %a)
  br i1 %cond, label %true, label %ret

true:
  %b = icmp ne i32 %arg2, -5
  call void @g2(i1 %b)
  br label %ret

ret:
  ret void
}
declare void @g2(i1)

; CHECK: Function: test_icmp_neg2
; CHECK: Hoist constant (i32 -500) to BB entry
define void @test_icmp_neg2(i1 %cond, i32 %arg, i32 %arg2) {
entry:
  %a = icmp ne i32 %arg, -500
  call void @g2(i1 %a)
  br i1 %cond, label %true, label %ret

true:
  %b = icmp ne i32 %arg2, -500
  call void @g2(i1 %b)
  br label %ret

ret:
  ret void
}

; CHECK: Function: test_add_neg
; CHECK-NOT: Collect constant i32 -5
define void @test_add_neg(i1 %cond, i32 %arg, i32 %arg2) {
entry:
  %a = add i32 %arg, -5
  call void @g(i32 %a)
  br i1 %cond, label %true, label %ret

true:
  %b = add i32 %arg2, -5
  call void @g(i32 %b)
  br label %ret

ret:
  ret void
}
