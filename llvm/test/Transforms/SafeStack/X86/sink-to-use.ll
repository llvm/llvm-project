; Test that unsafe alloca address calculation is done immediately before each use.
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s

define void @f() safestack {
entry:
  %x0 = alloca i32, align 4
  %x1 = alloca i32, align 4

; CHECK: %[[A:.*]] = getelementptr i8, ptr %{{.*}}, i32 -4
; CHECK: call void @use(ptr %[[A]])
  call void @use(ptr %x0)

; CHECK: %[[B:.*]] = getelementptr i8, ptr %{{.*}}, i32 -8
; CHECK: call void @use(ptr %[[B]])
  call void @use(ptr %x1)
  ret void
}

declare void @use(ptr)
