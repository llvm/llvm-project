; RUN: llc %s -mtriple=arm64-apple-darwin -o - | FileCheck %s

; CHECK-LABEL: ; -- Begin function foo
; CHECK: foo:
define hidden i32 @foo() {
  entry:
  ret i32 30
}
; CHECK: ; -- End function

; CHECK-LABEL: ; -- Begin function bar
; CHECK: bar:
define i32 @bar() {
  entry:
  ret i32 30
}
; CHECK: ; -- End function
