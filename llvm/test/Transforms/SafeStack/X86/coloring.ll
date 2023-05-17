; RUN: opt -safe-stack -safe-stack-coloring=1 -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -safe-stack-coloring=1 -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

define void @f() safestack {
entry:
; CHECK:  %[[USP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
; CHECK:  %[[USST:.*]] = getelementptr i8, ptr %[[USP]], i32 -16

  %x = alloca i32, align 4
  %x1 = alloca i32, align 4
  %x2 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr %x)

; CHECK:  %[[A1:.*]] = getelementptr i8, ptr %[[USP]], i32 -4
; CHECK:  call void @capture(ptr nonnull %[[A1]])

  call void @capture(ptr nonnull %x)
  call void @llvm.lifetime.end.p0(i64 4, ptr %x)
  call void @llvm.lifetime.start.p0(i64 4, ptr %x1)

; CHECK:  %[[B1:.*]] = getelementptr i8, ptr %[[USP]], i32 -4
; CHECK:  call void @capture(ptr nonnull %[[B1]])

  call void @capture(ptr nonnull %x1)
  call void @llvm.lifetime.end.p0(i64 4, ptr %x1)
  call void @llvm.lifetime.start.p0(i64 4, ptr %x2)

; CHECK:  %[[C1:.*]] = getelementptr i8, ptr %[[USP]], i32 -4
; CHECK:  call void @capture(ptr nonnull %[[C1]])

  call void @capture(ptr nonnull %x2)
  call void @llvm.lifetime.end.p0(i64 4, ptr %x2)
  ret void
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
declare void @capture(ptr)
