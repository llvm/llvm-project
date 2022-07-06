; RUN: opt < %s -S -loop-reduce | FileCheck %s

define void @testIVNext(i64* nocapture %a, i64 signext %m, i64 signext %n) {
entry:
  br label %for.body

for.body:
  %indvars.iv.prol = phi i64 [ %indvars.iv.next.prol, %for.body ], [ %m, %entry ]
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %uglygep138 = getelementptr i64, i64* %a, i64 %i
  store i64 55, i64* %uglygep138, align 4
  %indvars.iv.next.prol = add nuw nsw i64 %indvars.iv.prol, 1
  %i.next = add i64 %i, 1
  %i.cmp.not = icmp eq i64 %i.next, %n
  br i1 %i.cmp.not, label %for.exit, label %for.body

; CHECK: entry:
; CHECK: %0 = add i64 %n, %m
; CHECK-NOT : %indvars.iv.next.prol
; CHECK-NOT: %indvars.iv.prol
; CHECK: %indvars.iv.unr = phi i64 [ %0, %for.exit ]
for.exit:
  %indvars.iv.next.prol.lcssa = phi i64 [ %indvars.iv.next.prol, %for.body ]
  br label %exit

exit:
  %indvars.iv.unr = phi i64 [ %indvars.iv.next.prol.lcssa, %for.exit ]
  ret void
}

define void @testIV(i64* nocapture %a, i64 signext %m, i64 signext %n) {
entry:
  br label %for.body

for.body:
  %iv.prol = phi i64 [ %iv.next.prol, %for.body ], [ %m, %entry ]
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %uglygep138 = getelementptr i64, i64* %a, i64 %i
  store i64 55, i64* %uglygep138, align 4
  %iv.next.prol = add nuw nsw i64 %iv.prol, 1
  %i.next = add i64 %i, 1
  %i.cmp.not = icmp eq i64 %i.next, %n
  br i1 %i.cmp.not, label %for.exit, label %for.body

; CHECK: entry:
; CHECK: %0 = add i64 %n, %m
; CHECK: %1 = add i64 %0, -1
; CHECK-NOT: %iv.next.prol
; CHECK-NOT: %iv.prol
; CHECK: %indvars.iv.unr = phi i64 [ %1, %for.exit ]
for.exit:
  %iv.prol.lcssa = phi i64 [ %iv.prol, %for.body ]
  br label %exit
exit:
  %indvars.iv.unr = phi i64 [%iv.prol.lcssa, %for.exit]
  ret void
}

define void @testNonIndVarPhi() {
cont5820:
  br label %for.cond5821

for.cond5821:                                     ; preds = %cont5825, %cont5820
  %0 = phi i32 [ 0, %cont5825 ], [ 1, %cont5820 ]
  br label %cont5825

cont5825:                                         ; preds = %for.cond5821
  br i1 false, label %for.cond5821, label %for.cond6403

for.cond6403:                                     ; preds = %dead, %cont5825
  %1 = phi i32 [ %.lcssa221, %dead ], [ 0, %cont5825 ]
  br label %for.cond6418

for.cond6418:                                     ; preds = %cont6497, %for.cond6403
  %2 = phi i32 [ %0, %cont6497 ], [ %1, %for.cond6403 ]
  %3 = phi i64 [ 1, %cont6497 ], [ 0, %for.cond6403 ]
  %cmp6419 = icmp ule i64 %3, 0
  br i1 %cmp6419, label %cont6497, label %for.end6730

cont6497:                                         ; preds = %for.cond6418
  %conv6498 = sext i32 %2 to i64
  br label %for.cond6418

for.end6730:                                      ; preds = %for.cond6418
; Check that we don't make changes for phis which are not considered
; induction variables
; CHECK: %.lcssa221 = phi i32 [ %2, %for.cond6418 ]
  %.lcssa221 = phi i32 [ %2, %for.cond6418 ]
  ret void

dead:                                             ; No predecessors!
  br label %for.cond6403
}
