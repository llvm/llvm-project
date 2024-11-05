; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0  -scalar-evolution-classify-expressions=0  2>&1 | FileCheck %s

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

define void @epilogue2(i64 %count) {
; CHECK-LABEL: 'epilogue2'
; CHECK-NEXT:  Determining loop execution counts for: @epilogue2
; CHECK-NEXT:  Loop %epilogue: backedge-taken count is (-1 + %count.epilogue)
; CHECK-NEXT:  Loop %epilogue: constant max backedge-taken count is i64 8
; CHECK-NEXT:  Loop %epilogue: symbolic max backedge-taken count is (-1 + %count.epilogue)
; CHECK-NEXT:  Loop %epilogue: Trip multiple is 1
; CHECK-NEXT:  Loop %while.body: backedge-taken count is ((-8 + %count) /u 8)
; CHECK-NEXT:  Loop %while.body: constant max backedge-taken count is i64 2305843009213693951
; CHECK-NEXT:  Loop %while.body: symbolic max backedge-taken count is ((-8 + %count) /u 8)
; CHECK-NEXT:  Loop %while.body: Trip multiple is 1
entry:
  %cmp = icmp ugt i64 %count, 9
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

define void @slt(i16 %a, i16 %b, i1 %c) {
; CHECK-LABEL: 'slt'
; CHECK-NEXT:  Determining loop execution counts for: @slt
; CHECK-NEXT:  Loop %loop: backedge-taken count is (63 + (-1 * %count))
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 -32704
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is (63 + (-1 * %count))
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %b2

b1:
  %cmp1 = icmp slt i16 %a, 8
  br i1 %cmp1, label %preheader, label %exit

b2:
  %cmp2 = icmp slt i16 %b, 8
  br i1 %cmp2, label %preheader, label %exit

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ]
  br label %loop

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, 1
  %exitcond = icmp slt i16 %iv.next, 64
  br i1 %exitcond, label %loop, label %exit

exit:
  ret void
}

define void @ult(i16 %a, i16 %b, i1 %c) {
; CHECK-LABEL: 'ult'
; CHECK-NEXT:  Determining loop execution counts for: @ult
; CHECK-NEXT:  Loop %loop: backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 -2
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is (-1 + %count)
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %b2

b1:
  %cmp1 = icmp ult i16 %a, 8
  br i1 %cmp1, label %exit, label %preheader

b2:
  %cmp2 = icmp ult i16 %b, 8
  br i1 %cmp2, label %exit, label %preheader

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ]
  br label %loop

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, -1
  %exitcond = icmp eq i16 %iv.next, 0
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @sgt(i16 %a, i16 %b, i1 %c) {
; CHECK-LABEL: 'sgt'
; CHECK-NEXT:  Determining loop execution counts for: @sgt
; CHECK-NEXT:  Loop %loop: backedge-taken count is %count
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 32767
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is %count
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %b2

b1:
  %cmp1 = icmp sgt i16 %a, 8
  br i1 %cmp1, label %preheader, label %exit

b2:
  %cmp2 = icmp sgt i16 %b, 8
  br i1 %cmp2, label %preheader, label %exit

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ]
  br label %loop

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, -1
  %exitcond = icmp slt i16 %iv.next, 0
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}


define void @mixed(i16 %a, i16 %b, i1 %c) {
; CHECK-LABEL: 'mixed'
; CHECK-NEXT:  Determining loop execution counts for: @mixed
; CHECK-NEXT:  Loop %loop: backedge-taken count is (-1 + (-1 * %count) + (64 smax (1 + %count)))
; CHECK-NEXT:  Loop %loop: constant max backedge-taken count is i16 -32704
; CHECK-NEXT:  Loop %loop: symbolic max backedge-taken count is (-1 + (-1 * %count) + (64 smax (1 + %count)))
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
entry:
  br i1 %c, label %b1, label %b2

b1:
  %cmp1 = icmp slt i16 %a, 8
  br i1 %cmp1, label %preheader, label %exit

b2:
  %cmp2 = icmp ult i16 %b, 8
  br i1 %cmp2, label %preheader, label %exit

preheader:
  %count = phi i16 [ %a, %b1 ], [ %b, %b2 ]
  br label %loop

loop:
  %iv = phi i16 [ %iv.next, %loop ], [ %count, %preheader ]
  %iv.next = add i16 %iv, 1
  %exitcond = icmp slt i16 %iv.next, 64
  br i1 %exitcond, label %loop, label %exit

exit:
  ret void
}
