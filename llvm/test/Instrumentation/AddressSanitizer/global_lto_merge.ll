; RUN: opt < %s -passes=asan -S | FileCheck %s
; RUN: opt < %s "-passes=asan,constmerge" -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%struct = type { i64, i64 }

@a = private unnamed_addr constant %struct { i64 16, i64 16 }, align 8
@b = private unnamed_addr constant %struct { i64 16, i64 16 }, align 8

; CHECK: @a = {{.*}} %struct
; CHECK: @b = {{.*}} %struct

; CHECK: @llvm.compiler.used =
; CHECK-SAME: ptr @a
; CHECK-SAME: ptr @b

define i32 @main(i32, ptr nocapture readnone) {
  %3 = alloca %struct, align 8
  %4 = alloca %struct, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull %3, ptr @a, i64 16, i32 8, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull %4, ptr @b, i64 16, i32 8, i1 false)
  call void asm sideeffect "", "r,r,~{dirflag},~{fpsr},~{flags}"(ptr nonnull %3, ptr nonnull %4)
  ret i32 0
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32, i1)
