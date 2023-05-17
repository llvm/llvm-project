; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @_MergedGlobals = private global <{ i32, i32 }> <{ i32 3, i32 3 }>, align 4

@a = internal global i32 1

@b = internal global i32 2

@c = internal global i32 3

@d = internal global i32 3

@llvm.used = appending global [1 x ptr] [ptr @a], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr @b], section "llvm.metadata"

define void @use() {
  ; CHECK: load i32, ptr @a
  %x = load i32, ptr @a
  ; CHECK: load i32, ptr @b
  %y = load i32, ptr @b
  ; CHECK: load i32, ptr @_MergedGlobals
  %z1 = load i32, ptr @c
  ; CHECK: load i32, ptr getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 1)
  %z2 = load i32, ptr @d
  ret void
}
