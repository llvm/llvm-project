; RUN: llc < %s -asm-verbose=false | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: .globl foo
; CHECK: .type foo,@function
; CHECK-LABEL: foo:
; CHECK: .size foo,
define i32* @foo() {
  ret i32* @bar
}

; CHECK: .type bar,@object
; CHECK: .globl bar
; CHECK: .size bar, 4
@bar = global i32 2
