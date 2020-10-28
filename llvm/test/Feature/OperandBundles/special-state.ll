; RUN: opt -S -early-cse -earlycse-debug-hash < %s | FileCheck %s

; This test isn't directly related to EarlyCSE or varargs.  It is just
; using these as a vehicle for testing the correctness of
; haveSameSpecialState around operand bundles.

declare i32 @foo(...)

define i32 @f() {
; CHECK-LABEL: @f(
 entry:
; CHECK: %v0 = call i32 (...) @foo(
; CHECK: %v1 = call i32 (...) @foo(
; CHECK: %v = add i32 %v0, %v1
; CHECK: ret i32 %v

  %v0 = call i32 (...) @foo(i32 10) readonly [ "foo"(i32 20) ]
  %v1 = call i32 (...) @foo() readonly [ "foo"(i32 10, i32 20) ]
  %v = add i32 %v0, %v1
  ret i32 %v
}
