; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that the "returned" attribute works with i128 types.
; PR36128

target triple = "wasm32-unknown-unknown"

declare i128 @bar(i128 returned)

define i128 @foo(i128) {
  %r = tail call i128 @bar(i128 %0)
  ret i128 %r
}

; CHECK: .functype bar (i32, i64, i64) -> ()

; CHECK-LABEL: foo:
; CHECK-NEXT: .functype foo (i32, i64, i64) -> ()
