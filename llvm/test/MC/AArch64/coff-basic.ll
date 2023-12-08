; RUN: llc -mtriple aarch64-windows < %s | FileCheck %s

define i32 @foo() {
entry:
  ret i32 1
}

; CHECK: .globl foo
