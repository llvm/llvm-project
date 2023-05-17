; RUN: opt < %s -passes=inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

; Always prefer call site attribute over function attribute

define internal i32 @inner1() {
; CHECK: @inner1(
  ret i32 1
}

define i32 @outer1() {
; CHECK-LABEL: @outer1(
; CHECK: call

   %r = call i32 @inner1() noinline
   ret i32 %r
}

define internal i32 @inner2() alwaysinline {
; CHECK: @inner2(
  ret i32 1
}

define i32 @outer2() {
; CHECK-LABEL: @outer2(
; CHECK: call

   %r = call i32 @inner2() noinline
   ret i32 %r
}

define i32 @inner3() alwaysinline {
; CHECK: @inner3(
  ret i32 1
}

define i32 @outer3() {
; CHECK-LABEL: @outer3(
; CHECK: call

   %r = call i32 @inner3() noinline
   ret i32 %r
}

define i32 @inner4() noinline {
; CHECK: @inner4(
  ret i32 1
}

define i32 @outer4() {
; CHECK-LABEL: @outer4(
; CHECK-NOT: call
; CHECK: ret

   %r = call i32 @inner4() alwaysinline

   ret i32 %r
}

