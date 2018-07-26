; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @_MergedGlobals = private global <{ i32, i32 }> <{ i32 3, i32 3 }>, align 4

@a = internal global i32 1

@b = internal global i32 2

@c = internal global i32 3

@d = internal global i32 3

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32* @a to i8*)], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (i32* @b to i8*)], section "llvm.metadata"

define void @use() {
  ; CHECK: load i32, i32* @a
  %x = load i32, i32* @a
  ; CHECK: load i32, i32* @b
  %y = load i32, i32* @b
  ; CHECK: load i32, i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @_MergedGlobals, i32 0, i32 0)
  %z1 = load i32, i32* @c
  ; CHECK: load i32, i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @_MergedGlobals, i32 0, i32 1)
  %z2 = load i32, i32* @d
  ret void
}
