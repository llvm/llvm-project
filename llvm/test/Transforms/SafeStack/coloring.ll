; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

define void @f() safestack {
entry:
; CHECK:  %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK:  %[[USST:.*]] = getelementptr i8, i8* %[[USP]], i32 -16

  %x = alloca i32, align 4
  %x1 = alloca i32, align 4
  %x2 = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start(i64 4, i8* %0)

; CHECK:  %[[A1:.*]] = getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:  %[[A2:.*]] = bitcast i8* %[[A1]] to i32*
; CHECK:  call void @capture(i32* nonnull %[[A2]])

  call void @capture(i32* nonnull %x)
  call void @llvm.lifetime.end(i64 4, i8* %0)
  %1 = bitcast i32* %x1 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %1)

; CHECK:  %[[B1:.*]] = getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:  %[[B2:.*]] = bitcast i8* %[[B1]] to i32*
; CHECK:  call void @capture(i32* nonnull %[[B2]])

  call void @capture(i32* nonnull %x1)
  call void @llvm.lifetime.end(i64 4, i8* %1)
  %2 = bitcast i32* %x2 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %2)

; CHECK:  %[[C1:.*]] = getelementptr i8, i8* %[[USP]], i32 -4
; CHECK:  %[[C2:.*]] = bitcast i8* %[[C1]] to i32*
; CHECK:  call void @capture(i32* nonnull %[[C2]])

  call void @capture(i32* nonnull %x2)
  call void @llvm.lifetime.end(i64 4, i8* %2)
  ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)
declare void @capture(i32*)
