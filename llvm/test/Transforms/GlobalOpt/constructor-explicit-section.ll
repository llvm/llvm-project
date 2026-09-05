; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; Make sure we don't put a non-zero initializer into BSS

; CHECK: @_ZL3obj = internal global %struct.Holder zeroinitializer, section ".bss", align 8

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Holder = type { ptr }

@_ZL3obj = internal global %struct.Holder zeroinitializer, section ".bss", align 8
@_ZL5table = internal constant [1 x i32] [i32 7], align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_ZL4initv, ptr null }]
@llvm.compiler.used = appending global [1 x ptr] [ptr @_ZL3obj], section "llvm.metadata"

; Function Attrs: mustprogress nounwind uwtable
define internal void @_ZL4initv() #0 {
entry:
  store ptr @_ZL5table, ptr @_ZL3obj, align 8
  ret void
}

define dso_local noundef i32 @main() {
  ret i32 0
}
