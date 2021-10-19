; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s
; RUN: opt < %s "-passes=asan-pipeline,constmerge" -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%struct = type { i64, i64 }

@a = private unnamed_addr constant %struct { i64 16, i64 16 }, align 8
@b = private unnamed_addr constant %struct { i64 16, i64 16 }, align 8

; CHECK: @a = {{.*}} %struct
; CHECK: @b = {{.*}} %struct

; CHECK: @llvm.compiler.used =
; CHECK-SAME: i8* bitcast ({ %struct, [16 x i8] }* @a to i8*)
; CHECK-SAME: i8* bitcast ({ %struct, [16 x i8] }* @b to i8*)

define i32 @main(i32, i8** nocapture readnone) {
  %3 = alloca %struct, align 8
  %4 = alloca %struct, align 8
  %5 = bitcast %struct* %3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %5, i8* bitcast (%struct* @a to i8*), i64 16, i32 8, i1 false)
  %6 = bitcast %struct* %4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %6, i8* bitcast (%struct* @b to i8*), i64 16, i32 8, i1 false)
  call void asm sideeffect "", "r,r,~{dirflag},~{fpsr},~{flags}"(%struct* nonnull %3, %struct* nonnull %4)
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1)
