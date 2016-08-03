; Test that safestack layout reuses a region w/o fragmentation.
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

define void @f() safestack {
; CHECK-LABEL: define void @f
entry:
; CHECK:  %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -16

  %x0 = alloca i64, align 8
  %x1 = alloca i8, align 1
  %x2 = alloca i64, align 8

  %x0a = bitcast i64* %x0 to i8*
  %x2a = bitcast i64* %x2 to i8*

  call void @llvm.lifetime.start(i64 4, i8* %x0a)
  call void @capture64(i64* %x0)
  call void @llvm.lifetime.end(i64 4, i8* %x0a)

  call void @llvm.lifetime.start(i64 4, i8* %x1)
  call void @llvm.lifetime.start(i64 4, i8* %x2a)
  call void @capture8(i8* %x1)
  call void @capture64(i64* %x2)
  call void @llvm.lifetime.end(i64 4, i8* %x1)
  call void @llvm.lifetime.end(i64 4, i8* %x2a)

; Test that i64 allocas share space.
; CHECK: getelementptr i8, i8* %unsafe_stack_ptr, i32 -8
; CHECK: getelementptr i8, i8* %unsafe_stack_ptr, i32 -9
; CHECK: getelementptr i8, i8* %unsafe_stack_ptr, i32 -8

  ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)
declare void @capture8(i8*)
declare void @capture64(i64*)
