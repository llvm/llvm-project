; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0  -scalar-evolution-classify-expressions=0  2>&1 | FileCheck %s

define void @slt(i16 %a, i16 %b, i1 %c) {
; CHECK-LABEL: 'slt'
; CHECK-NEXT:  Determining loop execution counts for: @slt
; CHECK-NEXT:  Loop %loop: backedge-taken count is (19 + (-1 * %count)<nsw>)<nsw>
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 18
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is (19 + (-1 * %count)<nsw>)<nsw>
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %b2

b1:
  %cmp1 = icmp slt i16 %a, 1
  br i1 %cmp1, label %exit, label %preheader

b2:
  %cmp2 = icmp slt i16 %b, 4
  br i1 %cmp2, label %exit, label %preheader

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ]
  %cmp3 = icmp sle i16 %count, 19
  br i1 %cmp3, label %loop, label %exit

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, 1
  %exitcond = icmp eq i16 %iv.next, 20
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @ult(i16 %a, i16 %b, i1 %c) {
; CHECK-LABEL: 'ult'
; CHECK-NEXT:  Determining loop execution counts for: @ult
; CHECK-NEXT:  Loop %loop: backedge-taken count is (21 + (-1 * %count))
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 19
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is (21 + (-1 * %count))
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %b2

b1:
  %cmp1 = icmp ult i16 %a, 2
  br i1 %cmp1, label %exit, label %preheader

b2:
  %cmp2 = icmp ult i16 %b, 5
  br i1 %cmp2, label %exit, label %preheader

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ]
  %cmp3 = icmp ule i16 %count, 20
  br i1 %cmp3, label %loop, label %exit

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, 1
  %exitcond = icmp eq i16 %iv.next, 22
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @sgt(i16 %a, i16 %b, i1 %c) {
; CHECK-LABEL: 'sgt'
; CHECK-NEXT:  Determining loop execution counts for: @sgt
; CHECK-NEXT:  Loop %loop: backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 9
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %b2

b1:
  %cmp1 = icmp sgt i16 %a, 10
  br i1 %cmp1, label %exit, label %preheader

b2:
  %cmp2 = icmp sgt i16 %b, 8
  br i1 %cmp2, label %exit, label %preheader

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ]
  %cmp3 = icmp sge i16 %count, 1
  br i1 %cmp3, label %loop, label %exit

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, -1
  %exitcond = icmp eq i16 %iv.next, 0
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @ugt(i16 %a, i16 %b, i1 %c) {
; CHECK-LABEL: 'ugt'
; CHECK-NEXT:  Determining loop execution counts for: @ugt
; CHECK-NEXT:  Loop %loop: backedge-taken count is (-1 + %count)<nsw>
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 10
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is (-1 + %count)<nsw>
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %b2

b1:
  %cmp1 = icmp ugt i16 %a, 11
  br i1 %cmp1, label %exit, label %preheader

b2:
  %cmp2 = icmp ugt i16 %b, 7
  br i1 %cmp2, label %exit, label %preheader

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ]
  %cmp3 = icmp ne i16 %count, 0
  br i1 %cmp3, label %loop, label %exit

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, -1
  %exitcond = icmp eq i16 %iv.next, 0
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @three_incoming(i16 %a, i16 %b, i1 %c, i1 %d) {
; CHECK-LABEL: 'three_incoming'
; CHECK-NEXT:  Determining loop execution counts for: @three_incoming
; CHECK-NEXT:  Loop %loop: backedge-taken count is (-1 + %count)<nsw>
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 11
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is (-1 + %count)<nsw>
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %entry2

entry2:
  br i1 %d, label %b2, label %b3

b1:
  %cmp1 = icmp ugt i16 %a, 10
  br i1 %cmp1, label %exit, label %preheader

b2:
  %cmp2 = icmp ugt i16 %b, 8
  br i1 %cmp2, label %exit, label %preheader

b3:
  %cmp3 = icmp ugt i16 %b, 12
  br i1 %cmp3, label %exit, label %preheader

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ], [ %b, %b3 ]
  %cmp4 = icmp ne i16 %count, 0
  br i1 %cmp4, label %loop, label %exit

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, -1
  %exitcond = icmp eq i16 %iv.next, 0
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @mixed(i16 %a, i16 %b, i1 %c) {
; CHECK-LABEL: 'mixed'
; CHECK-NEXT:  Determining loop execution counts for: @mixed
; CHECK-NEXT:  Loop %loop: backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 -2
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %b2

b1:
  %cmp1 = icmp ugt i16 %a, 10
  br i1 %cmp1, label %exit, label %preheader

b2:
  %cmp2 = icmp sgt i16 %b, 8
  br i1 %cmp2, label %exit, label %preheader

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ]
  %cmp3 = icmp ne i16 %count, 0
  br i1 %cmp3, label %loop, label %exit

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, -1
  %exitcond = icmp eq i16 %iv.next, 0
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @one_constant(i16 %a, i16 %b, i1 %c, i16 %d) {
; CHECK-LABEL: 'one_constant'
; CHECK-NEXT:  Determining loop execution counts for: @one_constant
; CHECK-NEXT:  Loop %loop: backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 -2
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %b2

b1:
  %cmp1 = icmp ugt i16 %a, 10
  br i1 %cmp1, label %exit, label %preheader

b2:
  %cmp2 = icmp ugt i16 %b, %d
  br i1 %cmp2, label %exit, label %preheader

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ]
  %cmp3 = icmp ne i16 %count, 0
  br i1 %cmp3, label %loop, label %exit

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, -1
  %exitcond = icmp eq i16 %iv.next, 0
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @epilogue(i64 %count) {
; CHECK-LABEL: 'epilogue'
; CHECK-NEXT:  Determining loop execution counts for: @epilogue
; CHECK-NEXT:  Loop %epilogue: backedge-taken count is (-1 + %count.epilogue)
; CHECK-NEXT:  Loop %epilogue: constant max backedge-taken count is i64 6
; CHECK-NEXT:  Loop %epilogue: symbolic max backedge-taken count is (-1 + %count.epilogue)
; CHECK-NEXT:  Loop %epilogue: Trip multiple is 1
; CHECK-NEXT:  Loop %while.body: backedge-taken count is ((-8 + %count) /u 8)
; CHECK-NEXT:  Loop %while.body: constant max backedge-taken count is i64 2305843009213693951
; CHECK-NEXT:  Loop %while.body: symbolic max backedge-taken count is ((-8 + %count) /u 8)
; CHECK-NEXT:  Loop %while.body: Trip multiple is 1
entry:
  %cmp = icmp ugt i64 %count, 7
  br i1 %cmp, label %while.body, label %epilogue.preheader

while.body:
  %iv = phi i64 [ %sub, %while.body ], [ %count, %entry ]
  %sub = add i64 %iv, -8
  %exitcond.not = icmp ugt i64 %sub, 7
  br i1 %exitcond.not, label %while.body, label %while.loopexit

while.loopexit:
  %sub.exit = phi i64 [ %sub, %while.body ]
  br label %epilogue.preheader

epilogue.preheader:
  %count.epilogue = phi i64 [ %count, %entry ], [ %sub.exit, %while.loopexit ]
  %epilogue.cmp = icmp eq i64 %count.epilogue, 0
  br i1 %epilogue.cmp, label %exit, label %epilogue

epilogue:
  %iv.epilogue = phi i64 [ %dec, %epilogue ], [ %count.epilogue, %epilogue.preheader ]
  %dec = add i64 %iv.epilogue, -1
  %exitcond.epilogue = icmp eq i64 %dec, 0
  br i1 %exitcond.epilogue, label %exit, label %epilogue

exit:
  ret void
}

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

; Checks correct traversal for loops without a unique predecessor
; outside the loop.
define void @pr120615() {
; CHECK-LABEL: pr120615
; CHECK-NEXT:  Determining loop execution counts for: @pr120615
; CHECK-NEXT:  Loop %header: backedge-taken count is i32 0
; CHECK-NEXT:  Loop %header: constant max backedge-taken count is i32 0
; CHECK-NEXT:  Loop %header: symbolic max backedge-taken count is i32 0
; CHECK-NEXT:  Loop %header: Trip multiple is 1
entry:
  br label %header

bb:
  br label %header

header:
  %0 = phi i32 [ %1, %header ], [ 0, %bb ], [ 0, %entry ]
  %1 = add i32 %0, 1
  %icmp = icmp slt i32 %0, 0
  br i1 %icmp, label %header, label %exit

exit:
  ret void

}
