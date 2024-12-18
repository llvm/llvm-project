; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0  -scalar-evolution-classify-expressions=0  2>&1 | FileCheck %s

declare void @llvm.assume(i1)

define void @pr120442() {
; CHECK-LABEL: 'pr120442'
; CHECK-NEXT:  Determining loop execution counts for: @pr120442
; CHECK-NEXT:  Loop %bb2: backedge-taken count is i32 0
; CHECK-NEXT:  Loop %bb2: constant max backedge-taken count is i32 0
; CHECK-NEXT:  Loop %bb2: symbolic max backedge-taken count is i32 0
; CHECK-NEXT:  Loop %bb2: Trip multiple is 1
; CHECK-NEXT:  Loop %bb1: <multiple exits> Unpredictable backedge-taken count.
; CHECK-NEXT:  Loop %bb1: Unpredictable constant max backedge-taken count.
; CHECK-NEXT:  Loop %bb1: Unpredictable symbolic max backedge-taken count.
bb:
  call void @llvm.assume(i1 false)
  call void @llvm.assume(i1 false)
  br label %bb6

bb1:
  br label %bb2

bb2:
  %phi = phi i32 [ %add, %bb2 ], [ 0, %bb1 ]
  %add = add i32 %phi, 1
  %icmp = icmp ugt i32 %add, 0
  br i1 %icmp, label %bb1, label %bb2

bb5:
  br i1 false, label %bb6, label %bb5

bb6:
  %phi7 = phi i32 [ 0, %bb5 ], [ 0, %bb ]
  br label %bb1
}
