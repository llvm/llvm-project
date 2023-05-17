; RUN: llvm-extract -S --bb=func:bb4 -aggregate-extracted-args=0 < %s | FileCheck %s

; FIXME: aggregate-extracted-args doesn't work for other reasons
; XUN: llvm-extract -S --bb=func:bb4 -aggregate-extracted-args=1 < %s | FileCheck %s

target datalayout = "A5-G1-ni:7"

; Check that there's no assert from incorrect pointer types used in the new arguments.

; CHECK-LABEL: define dso_local void @func.bb4(i32 %orig.arg.0, ptr addrspace(5) %tmp1.out, ptr addrspace(5) %add.out) {
; CHECK: bb4:
; CHECK-NEXT: %tmp0 = add i32 0, 0
; CHECK-NEXT: %tmp1 = add i32 1, 1
; CHECK-NEXT: store i32 %tmp1, ptr addrspace(5) %tmp1.out, align 4
; CHECK-NEXT: %add = add i32 %tmp0, %orig.arg.0
; CHECK-NEXT: store i32 %add, ptr addrspace(5) %add.out, align 4
; CHECK-NEXT: br label %bb5.exitStub
define void @func(i32 %orig.arg.0, ptr addrspace(1) %orig.arg.1) {
bb:
  br label %bb4

bb4:                                              ; preds = %bb
  %tmp0 = add i32 0, 0
  %tmp1 = add i32 1, 1
  %add = add i32 %tmp0, %orig.arg.0
  br label %bb5

bb5:                                              ; preds = %bb5, %bb4
  %tmp6 = phi i32 [ %add, %bb4 ], [ 0, %bb5 ]
  %tmp7 = phi i32 [ %tmp1, %bb4 ], [ 2, %bb5 ]
  br label %bb5
}
