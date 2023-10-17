; RUN: opt < %s -passes=globalopt -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".global foo1"
module asm "foo1: jmp bar1"
module asm ".global foo2"
module asm "foo2: jmp bar2"

; The `llvm.compiler.used` indicates that `foo1` and `foo2` have associated symbol references in asm.
; Checking globalopt does not remove these two symbols.
@llvm.compiler.used = appending global [2 x ptr] [ptr @bar1, ptr @bar2], section "llvm.metadata"
; CHECK: @llvm.compiler.used = appending global [2 x ptr] [ptr @bar1, ptr @bar2], section "llvm.metadata"

@bar2 = internal alias void (), ptr @bar1
; CHECK: @bar2 = internal alias void (), ptr @bar1

define internal void @bar1() {
; CHECK: define internal void @bar1()
  ret void
}
