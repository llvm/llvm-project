; RUN: opt -inline < %s -S -o - -inline-threshold=10 | FileCheck %s

target datalayout = "p:32:32"

define i32 @outer1() {
; CHECK: @outer1
; CHECK-NOT: call
; CHECK: ret i32

  %ptr = alloca i32
  %ptr1 = getelementptr inbounds i32* %ptr, i32 0
  %ptr2 = getelementptr inbounds i32* %ptr, i32 42
  %result = call i32 @inner1(i32* %ptr1, i32* %ptr2)
  ret i32 %result
}

define i32 @inner1(i32* %begin, i32* %end) {
  %begin.i = ptrtoint i32* %begin to i32
  %end.i = ptrtoint i32* %end to i32
  %distance = sub i32 %end.i, %begin.i
  %icmp = icmp sle i32 %distance, 42
  br i1 %icmp, label %then, label %else

then:
  ret i32 3

else:
  %t = load i32* %begin
  ret i32 %t
}

define i32 @outer2(i32* %ptr) {
; Test that an inbounds GEP disables this -- it isn't safe in general as
; wrapping changes the behavior of lessthan and greaterthan comparisions.
; CHECK: @outer2
; CHECK: call i32 @inner2
; CHECK: ret i32

  %ptr1 = getelementptr i32* %ptr, i32 0
  %ptr2 = getelementptr i32* %ptr, i32 42
  %result = call i32 @inner2(i32* %ptr1, i32* %ptr2)
  ret i32 %result
}

define i32 @inner2(i32* %begin, i32* %end) {
  %begin.i = ptrtoint i32* %begin to i32
  %end.i = ptrtoint i32* %end to i32
  %distance = sub i32 %end.i, %begin.i
  %icmp = icmp sle i32 %distance, 42
  br i1 %icmp, label %then, label %else

then:
  ret i32 3

else:
  %t = load i32* %begin
  ret i32 %t
}
