; RUN: opt -passes=inline < %s -S -o - -inline-threshold=10 | FileCheck %s

target datalayout = "p:32:32-p1:64:64-p2:16:16-n16:32:64"

define i32 @outer1() {
; CHECK-LABEL: @outer1(
; CHECK-NOT: call i32
; CHECK: ret i32

  %ptr = alloca i32
  %ptr2 = getelementptr inbounds i32, ptr %ptr, i32 42
  %result = call i32 @inner1(ptr %ptr, ptr %ptr2)
  ret i32 %result
}

define i32 @inner1(ptr %begin, ptr %end) {
  call void @extern()
  %begin.i = ptrtoint ptr %begin to i32
  %end.i = ptrtoint ptr %end to i32
  %distance = sub i32 %end.i, %begin.i
  %icmp = icmp sle i32 %distance, 42
  br i1 %icmp, label %then, label %else

then:
  ret i32 3

else:
  %t = load i32, ptr %begin
  ret i32 %t
}

define i32 @outer1_as1(ptr addrspace(1) %ptr) {
; CHECK-LABEL: @outer1_as1(
; CHECK-NOT: call
; CHECK: ret i32
  %ptr2 = getelementptr inbounds i32, ptr addrspace(1) %ptr, i32 42
  %result = call i32 @inner1_as1(ptr addrspace(1) %ptr, ptr addrspace(1) %ptr2)
  ret i32 %result
}

; Make sure that the address space's larger size makes the ptrtoints
; not no-ops preventing inlining
define i32 @inner1_as1(ptr addrspace(1) %begin, ptr addrspace(1) %end) {
  %begin.i = ptrtoint ptr addrspace(1) %begin to i32
  %end.i = ptrtoint ptr addrspace(1) %end to i32
  %distance = sub i32 %end.i, %begin.i
  %icmp = icmp sle i32 %distance, 42
  br i1 %icmp, label %then, label %else

then:
  ret i32 3

else:
  %t = load i32, ptr addrspace(1) %begin
  ret i32 %t
}

define i32 @outer2(ptr %ptr) {
; Test that an inbounds GEP disables this -- it isn't safe in general as
; wrapping changes the behavior of lessthan and greaterthan comparisons.
; CHECK-LABEL: @outer2(
; CHECK: call i32 @inner2
; CHECK: ret i32

  %ptr2 = getelementptr i32, ptr %ptr, i32 42
  %result = call i32 @inner2(ptr %ptr, ptr %ptr2)
  ret i32 %result
}

define i32 @inner2(ptr %begin, ptr %end) {
  call void @extern()
  %begin.i = ptrtoint ptr %begin to i32
  %end.i = ptrtoint ptr %end to i32
  %distance = sub i32 %end.i, %begin.i
  %icmp = icmp sle i32 %distance, 42
  br i1 %icmp, label %then, label %else

then:
  ret i32 3

else:
  %t = load i32, ptr %begin
  ret i32 %t
}

define i32 @outer3(ptr addrspace(1) %ptr) {
; CHECK-LABEL: @outer3(
; CHECK-NOT: call i32
; CHECK: ret i32 3
; CHECK-LABEL: @inner3(
  %result = call i32 @inner3(ptr addrspace(1) %ptr)
  ret i32 %result
}

define i32 @inner3(ptr addrspace(1) %ptr) {
  call void @extern()
  %ptr.i = ptrtoint ptr addrspace(1) %ptr to i64
  %distance = sub i64 %ptr.i, %ptr.i
  %icmp = icmp eq i64 %distance, 0
  br i1 %icmp, label %then, label %else

then:
  ret i32 3

else:
  ret i32 5
}


; The inttoptrs are free since it is a smaller integer to a larger
; pointer size
define i32 @inttoptr_free_cost(i32 %a, i32 %b, i32 %c) {
  call void @extern()
  %p1 = inttoptr i32 %a to ptr addrspace(1)
  %p2 = inttoptr i32 %b to ptr addrspace(1)
  %p3 = inttoptr i32 %c to ptr addrspace(1)
  %t1 = load i32, ptr addrspace(1) %p1
  %t2 = load i32, ptr addrspace(1) %p2
  %t3 = load i32, ptr addrspace(1) %p3
  %s = add i32 %t1, %t2
  %s1 = add i32 %s, %t3
  ret i32 %s1
}

define i32 @inttoptr_free_cost_user(i32 %begin, i32 %end) {
; CHECK-LABEL: @inttoptr_free_cost_user(
; CHECK-NOT: call i32
  %x = call i32 @inttoptr_free_cost(i32 %begin, i32 %end, i32 9)
  ret i32 %x
}

; The inttoptrs have a cost since it is a larger integer to a smaller
; pointer size
define i32 @inttoptr_cost_smaller_ptr(i32 %a, i32 %b, i32 %c) {
  call void @extern()
  %p1 = inttoptr i32 %a to ptr addrspace(2)
  %p2 = inttoptr i32 %b to ptr addrspace(2)
  %p3 = inttoptr i32 %c to ptr addrspace(2)
  %t1 = load i32, ptr addrspace(2) %p1
  %t2 = load i32, ptr addrspace(2) %p2
  %t3 = load i32, ptr addrspace(2) %p3
  %s = add i32 %t1, %t2
  %s1 = add i32 %s, %t3
  ret i32 %s1
}

define i32 @inttoptr_cost_smaller_ptr_user(i32 %begin, i32 %end) {
; CHECK-LABEL: @inttoptr_cost_smaller_ptr_user(
; CHECK: call i32
  %x = call i32 @inttoptr_cost_smaller_ptr(i32 %begin, i32 %end, i32 9)
  ret i32 %x
}

declare void @extern()
