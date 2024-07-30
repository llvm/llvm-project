; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s
; RUN: opt -passes='global-merge<max-offset=100>' -S -o - %s | FileCheck %s

; NOTE: This is a copy of the llvm/test/Transforms/GlobalMerge/used.ll test,
; using `private` global variables instead of `internal`. This is to show that
; that private globals can be merged in the GlobalMerge pass.

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @_MergedGlobals = private global <{ i32, i32 }> <{ i32 3, i32 3 }>, align 4

@a = private global i32 1
@b = private global i32 2
@c = private global i32 3
@d = private global i32 3

@llvm.used = appending global [1 x ptr] [ptr @a], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr @b], section "llvm.metadata"

define void @use_private() {
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
