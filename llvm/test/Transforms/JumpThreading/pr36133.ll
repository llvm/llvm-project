; RUN: opt -passes=jump-threading -S < %s | FileCheck %s
@global = external global ptr, align 8

define i32 @foo(i32 %arg) {
; CHECK-LABEL: @foo
; CHECK-LABEL: bb:
; CHECK: icmp eq
; CHECK-NEXT: br i1 %tmp1, label %bb7, label %bb7
bb:
  %tmp = load ptr, ptr @global, align 8
  %tmp1 = icmp eq ptr %tmp, null
  br i1 %tmp1, label %bb3, label %bb2

; CHECK-NOT: bb2:
bb2:
  br label %bb3

; CHECK-NOT: bb3:
bb3:
  %tmp4 = phi i8 [ 1, %bb2 ], [ 0, %bb ]
  %tmp5 = icmp eq i8 %tmp4, 0
  br i1 %tmp5, label %bb7, label %bb6

; CHECK-NOT: bb6:
bb6:
  br label %bb7

; CHECK-LABEL: bb7:
bb7:
  %tmp8 = icmp eq i32 %arg, -1
  br i1 %tmp8, label %bb9, label %bb10

; CHECK-LABEL: bb9:
bb9:
  ret i32 0

; CHECK-LABEL: bb10:
bb10:
  %tmp11 = icmp sgt i32 %arg, -1
  call void @llvm.assume(i1 %tmp11)
  ret i32 1
}

declare void @llvm.assume(i1)
