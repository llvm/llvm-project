; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

; CHECK: __safestack_unsafe_stack_ptr
define void @oob_read(i8* %ptr) safestack {
  %1 = alloca i8
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %ptr, i8* align 1 %1, i64 4, i1 false)
  ret void
}
