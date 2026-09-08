; RUN: opt -passes=argpromotion -S < %s | FileCheck %s

; ArgumentPromotion promoting a by-ref argument of a `noipa` function.
; If `noipa` is working, argpromotion should NOT promote the pointer argument,
; because it requires inspecting the function body to determine load patterns.

; CHECK: define internal i32 @foo(i32 %p.0.val)
define internal i32 @foo(ptr %p) noinline {
  %v = load i32, ptr %p
  ret i32 %v
}

; CHECK: define internal i32 @foo_noipa(ptr %p)
define internal i32 @foo_noipa(ptr %p) noipa noinline {
  %v = load i32, ptr %p
  ret i32 %v
}

define i32 @bar() {
  %a = alloca i32
  store i32 7, ptr %a
  %r = call i32 @foo(ptr %a)
  %b = alloca i32
  store i32 %r, ptr %b
  %s = call i32 @foo_noipa(ptr %b)
  ret i32 %s
}
