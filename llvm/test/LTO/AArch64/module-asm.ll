; RUN: llvm-as %s -o %t.o
; RUN: llvm-lto %t.o --list-symbols-only | FileCheck %s

; CHECK: ___foo    { function defined hidden }
; CHECK: ___bar    { function defined default }
; CHECK: _foo    { data defined default }
; CHECK: ___foo    { asm extern }

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx12.0.0"

module asm ".globl _foo"
module asm "_foo = ___foo"

define hidden i32 @__foo() {
  ret i32 0
}

define i32 @__bar() {
  ret i32 0
}
