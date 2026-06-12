; Test: MergeFunctions merging a noipa function with an identical one.
; If noipa is working, MergeFunctions should NOT merge @foo into @baz
; (or vice versa), because it inspects the function definitions.
;
; RUN: opt -passes=mergefunc -S < %s | FileCheck %s

; CHECK: define internal i32 @foo(
define internal i32 @foo(i32 %x, i32 %y) noipa noinline {
  %sum = add i32 %x, %y
  %mul = mul i32 %sum, %x
  %sub = sub i32 %mul, %y
  ret i32 %sub
}

; CHECK: define internal i32 @baz(
define internal i32 @baz(i32 %x, i32 %y) noipa noinline {
  %sum = add i32 %x, %y
  %mul = mul i32 %sum, %x
  %sub = sub i32 %mul, %y
  ret i32 %sub
}

define i32 @bar() {
  %r1 = call i32 @foo(i32 1, i32 2)
  %r2 = call i32 @baz(i32 3, i32 4)
  %r = add i32 %r1, %r2
  ret i32 %r
}
