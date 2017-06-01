; RUN: opt -S -loop-unswitch < %s | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define void @f_0(i32 %n, i32* %ptr, i1 %c) {
; CHECK-LABEL: @f_0(
; CHECK: loop.us:
; CHECK-NOT: guard
; CHECK: loop:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"() ]
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1
  call void(i1, ...) @llvm.experimental.guard(i1 %c) [ "deopt"() ]
  store volatile i32 0, i32* %ptr
  %becond = icmp ult i32 %iv.inc, %n
  br i1 %becond, label %leave, label %loop

leave:
  ret void
}

define void @f_1(i32 %n, i32* %ptr, i1 %c_0, i1 %c_1) {
; CHECK-LABEL: @f_1(
; CHECK: loop.us.us:
; CHECK-NOT: guard
; CHECK: loop.us:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"(i32 2) ]
; CHECK-NOT: guard
; CHECK: loop.us1:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"(i32 1) ]
; CHECK-NOT: guard
; CHECK: loop:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"(i32 1) ]
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"(i32 2) ]
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1
  call void(i1, ...) @llvm.experimental.guard(i1 %c_0) [ "deopt"(i32 1) ]
  store volatile i32 0, i32* %ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c_1) [ "deopt"(i32 2) ]
  %becond = icmp ult i32 %iv.inc, %n
  br i1 %becond, label %leave, label %loop

leave:
  ret void
}

; Basic negative test

define void @f_3(i32 %n, i32* %ptr, i1* %c_ptr) {
; CHECK-LABEL: @f_3(
; CHECK-NOT: loop.us:
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1
  %c = load volatile i1, i1* %c_ptr
  call void(i1, ...) @llvm.experimental.guard(i1 %c) [ "deopt"() ]
  store volatile i32 0, i32* %ptr
  %becond = icmp ult i32 %iv.inc, %n
  br i1 %becond, label %leave, label %loop

leave:
  ret void
}

define void @f_4(i32 %n, i32* %ptr, i1 %c) {
; CHECK-LABEL: @f_4(
;
; Demonstrate that unswitching on one guard can cause another guard to
; be erased (this has implications on what guards we can keep raw
; pointers to).
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1
  call void(i1, ...) @llvm.experimental.guard(i1 %c) [ "deopt"(i32 1) ]
  store volatile i32 0, i32* %ptr
  %neg = xor i1 %c, 1
  call void(i1, ...) @llvm.experimental.guard(i1 %neg) [ "deopt"(i32 2) ]
  %becond = icmp ult i32 %iv.inc, %n
  br i1 %becond, label %leave, label %loop

leave:
  ret void
}
