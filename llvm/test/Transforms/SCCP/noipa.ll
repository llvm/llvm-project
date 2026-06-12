; Test: IPSCCP constant-propagating through arguments of a noipa function.
; If noipa is working, IPSCCP should NOT propagate the constant 42 into @foo,
; because analyzing calls to @foo to determine argument values is IPA.
;
; RUN: opt -passes=ipsccp -S < %s | FileCheck %s

; CHECK:      define internal i32 @foo(i32 %x)
; CHECK-NEXT:   ret i32 %x
define internal i32 @foo(i32 %x) noipa noinline {
  ret i32 %x
}

define i32 @bar() {
  %r = call i32 @foo(i32 42)
  ret i32 %r
}
