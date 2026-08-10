; RUN: llc < %s -asm-verbose=false | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK: .globl foo
; CHECK: .type foo,@function
; CHECK-LABEL: foo:
; CHECK: end_function
define ptr @foo() {
  ret ptr @bar
}

; CHECK: .type bar,@object
; CHECK: .globl bar
; CHECK: .size bar, 4
@bar = global i32 2
