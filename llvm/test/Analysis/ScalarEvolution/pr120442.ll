; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0  -scalar-evolution-classify-expressions=0  2>&1 | FileCheck %s

declare void @llvm.assume(i1)

; Checks that the presence of assumptions does not interfere with
; exiting loop guard collection via following loop predecessors.
define void @pr120442(i1 %c.1, i1 %c.2) {
; CHECK-LABEL: 'pr120442'
; CHECK-NEXT:  Determining loop execution counts for: @pr120442
; CHECK-NEXT:  Loop %inner.header: backedge-taken count is i32 0
; CHECK-NEXT:  Loop %inner.header: constant max backedge-taken count is i32 0
; CHECK-NEXT:  Loop %inner.header: symbolic max backedge-taken count is i32 0
; CHECK-NEXT:  Loop %inner.header: Trip multiple is 1
entry:
  call void @llvm.assume(i1 %c.1)
  call void @llvm.assume(i1 %c.2)
  br label %outer.header

outer.header:
  %phi7 = phi i32 [ 0, %bb ], [ 0, %entry ]
  br label %inner.header

bb:
  br i1 false, label %outer.header, label %bb

inner.header:
  %phi = phi i32 [ %add, %inner.header ], [ 0, %outer.header ]
  %add = add i32 %phi, 1
  %icmp = icmp ugt i32 %add, 0
  br i1 %icmp, label %exit, label %inner.header

exit:
  ret void
}
