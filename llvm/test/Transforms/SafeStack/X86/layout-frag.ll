; Test that safestack layout reuses a region w/o fragmentation.
; RUN: opt -safe-stack -safe-stack-coloring=1 -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

define void @f() safestack {
; CHECK-LABEL: define void @f
entry:
; CHECK:  %[[USP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
; CHECK:   getelementptr i8, ptr %[[USP]], i32 -16

  %x0 = alloca i64, align 8
  %x1 = alloca i8, align 1
  %x2 = alloca i64, align 8


  call void @llvm.lifetime.start.p0(i64 8, ptr %x0)
  call void @capture64(ptr %x0)
  call void @llvm.lifetime.end.p0(i64 8, ptr %x0)

  call void @llvm.lifetime.start.p0(i64 1, ptr %x1)
  call void @llvm.lifetime.start.p0(i64 8, ptr %x2)
  call void @capture8(ptr %x1)
  call void @capture64(ptr %x2)
  call void @llvm.lifetime.end.p0(i64 1, ptr %x1)
  call void @llvm.lifetime.end.p0(i64 8, ptr %x2)

; Test that i64 allocas share space.
; CHECK: getelementptr i8, ptr %unsafe_stack_ptr, i32 -8
; CHECK: getelementptr i8, ptr %unsafe_stack_ptr, i32 -9
; CHECK: getelementptr i8, ptr %unsafe_stack_ptr, i32 -8

  ret void
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
declare void @capture8(ptr)
declare void @capture64(ptr)
