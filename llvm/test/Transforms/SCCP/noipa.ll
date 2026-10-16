; RUN: opt -passes=ipsccp -S < %s | FileCheck %s

; IPSCCP constant-propagating through arguments of a `noipa` function.
; If `noipa` is working, IPSCCP should NOT propagate the constant 42 into @foo,
; because analyzing calls to @foo to determine argument values is IPA.

; CHECK:      define internal i32 @foo(i32 %x)
; CHECK-NEXT:   ret i32 poison
define internal i32 @foo(i32 %x) noinline {
  ret i32 %x
}

; CHECK:      define internal i32 @foo_noipa(i32 %x)
; CHECK-NEXT:   ret i32 %x
define internal i32 @foo_noipa(i32 %x) noipa noinline {
  ret i32 %x
}

define i32 @bar() {
  %r = call i32 @foo(i32 42)
  %s = call i32 @foo_noipa(i32 42)
  ; CHECK: %res = add i32 42, %s
  %res = add i32 %r, %s
  ret i32 %res
}
