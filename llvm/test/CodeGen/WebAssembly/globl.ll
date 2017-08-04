; RUN: llc < %s -asm-verbose=false | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; CHECK: .globl foo
; CHECK-LABEL: foo:
define void @foo() {
  ret void
}

; Check import directives - must be at the end of the file
; CHECK: .import_global bar{{$}}
@bar = external global i32
