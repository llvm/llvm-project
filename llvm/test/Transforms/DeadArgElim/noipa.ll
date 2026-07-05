; RUN: opt -passes=deadargelim -S < %s | FileCheck %s

; DeadArgumentElimination removing unused args of a `noipa` function.
; If `noipa` is working, DAE should NOT remove the unused argument %y from @foo,
; because determining which arguments are dead requires inspecting the
; definition.

; CHECK: define internal i32 @foo(i32 %x)
define internal i32 @foo(i32 %x, i32 %y) noinline {
  ret i32 %x
}

; CHECK: define internal i32 @foo_noipa(i32 %x, i32 %y)
define internal i32 @foo_noipa(i32 %x, i32 %y) noipa noinline {
  ret i32 %x
}

define i32 @bar() {
  %r = call i32 @foo(i32 1, i32 2)
  %s = call i32 @foo_noipa(i32 1, i32 2)
  %res = add i32 %r, %s
  ret i32 %res
}
