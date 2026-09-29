; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; Test that when all members of llvm.compiler.used are found to be redundant
; we delete it instead of crashing.

define void @foo() {
  ret void
}

@llvm.used = appending global [1 x ptr] [ptr @foo], section "llvm.metadata"

@llvm.compiler.used = appending global [1 x ptr] [ptr @foo], section "llvm.metadata"

; CHECK-NOT: @llvm.compiler.used
; CHECK: @llvm.used = appending global [1 x ptr] [ptr @foo], section "llvm.metadata"
; CHECK-NOT: @llvm.compiler.used
