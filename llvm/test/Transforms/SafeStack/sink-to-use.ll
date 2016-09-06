; Test that unsafe alloca address calculation is done immediately before each use.
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s

define void @f() safestack {
entry:
  %x0 = alloca i32, align 4
  %x1 = alloca i32, align 4

; CHECK: %[[A:.*]] = getelementptr i8, i8* %{{.*}}, i32 -4
; CHECK: %[[X0:.*]] = bitcast i8* %[[A]] to i32*
; CHECK: call void @use(i32* %[[X0]])
  call void @use(i32* %x0)

; CHECK: %[[B:.*]] = getelementptr i8, i8* %{{.*}}, i32 -8
; CHECK: %[[X1:.*]] = bitcast i8* %[[B]] to i32*
; CHECK: call void @use(i32* %[[X1]])
  call void @use(i32* %x1)
  ret void
}

declare void @use(i32*)
