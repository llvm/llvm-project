; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

define void @f(i1 %d1, i1 %d2) safestack {
entry:
; CHECK-LABEL: define void @f(
; CHECK:         %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT:    getelementptr i8, i8* %[[USP]], i32 -16
; CHECK:         br i1 %d1, label %[[BB0:.*]], label %[[BB1:.*]]
  %a = alloca i32, align 8
  %b = alloca i32, align 8
  br i1 %d1, label %bb0, label %bb1

bb0:
; CHECK: [[BB0]]:
; CHECK: %[[Ai8:.*]] = getelementptr i8, i8* %unsafe_stack_ptr, i32
; CHECK: %[[AUNSAFE:.*]] = bitcast i8* %[[Ai8]] to i32*
; CHECK: br i1
  br i1 %d2, label %bb2, label %bb2

bb1:
; CHECK: [[BB1]]:
; CHECK: %[[Bi8:.*]] = getelementptr i8, i8* %unsafe_stack_ptr, i32
; CHECK: %[[BUNSAFE:.*]] = bitcast i8* %[[Bi8]] to i32*
; CHECK: br label
  br label %bb2

bb2:
; CHECK: phi i32* [ %[[AUNSAFE]], %[[BB0]] ], [ %[[AUNSAFE]], %[[BB0]] ], [ %[[BUNSAFE]], %[[BB1]] ]
  %c = phi i32* [ %a, %bb0 ], [ %a, %bb0 ], [ %b, %bb1 ]
  call void @capture(i32* %c)
  ret void
}

declare void @capture(i32*)
