; RUN: opt -passes=mergefunc -S < %s | FileCheck %s

; MergeFunctions merging a `noipa` function with an identical one.
; If `noipa` is working, MergeFunctions should NOT merge
; @foo_noipa into @baz_noipa (or vice versa), because it inspects
; the function definitions.

; CHECK-NOT: define internal i32 @foo(
define internal i32 @foo(i32 %x, i32 %y) noinline {
  %sum = add i32 %x, %y
  %mul = mul i32 %sum, %x
  %sub = sub i32 %mul, %y
  ret i32 %sub
}

; CHECK: define internal i32 @baz(
define internal i32 @baz(i32 %x, i32 %y) noinline {
  %sum = add i32 %x, %y
  %mul = mul i32 %sum, %x
  %sub = sub i32 %mul, %y
  ret i32 %sub
}

; CHECK: define internal i32 @foo_noipa(
define internal i32 @foo_noipa(i32 %x, i32 %y) noipa noinline {
  %sum = add i32 %x, %y
  %mul = mul i32 %sum, %x
  %sub = sub i32 %mul, %y
  ret i32 %sub
}

; CHECK: define internal i32 @baz_noipa(
define internal i32 @baz_noipa(i32 %x, i32 %y) noipa noinline {
  %sum = add i32 %x, %y
  %mul = mul i32 %sum, %x
  %sub = sub i32 %mul, %y
  ret i32 %sub
}

define i32 @bar() {
  ; CHECK: call i32 @baz(
  %r1 = call i32 @foo(i32 1, i32 2)
  ; CHECK: call i32 @baz(
  %r2 = call i32 @baz(i32 3, i32 4)
  ; CHECK: call i32 @foo_noipa(
  %r3 = call i32 @foo_noipa(i32 1, i32 2)
  ; CHECK: call i32 @baz_noipa(
  %r4 = call i32 @baz_noipa(i32 3, i32 4)
  %res1 = add i32 %r1, %r2
  %res2 = add i32 %r3, %r4
  %res3 = add i32 %res1, %res2
  ret i32 %res3
}
