; Test: ArgumentPromotion promoting a by-ref argument of a noipa function.
; If noipa is working, argpromotion should NOT promote the pointer argument,
; because it requires inspecting the function body to determine load patterns.
;
; RUN: opt -passes=argpromotion -S < %s | FileCheck %s

; CHECK: define internal i32 @foo(ptr %p)
define internal i32 @foo(ptr %p) noipa noinline {
  %v = load i32, ptr %p
  ret i32 %v
}

define i32 @bar() {
  %a = alloca i32
  store i32 7, ptr %a
  %r = call i32 @foo(ptr %a)
  ret i32 %r
}
