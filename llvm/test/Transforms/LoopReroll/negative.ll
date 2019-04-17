; RUN: opt -S  -loop-reroll   %s | FileCheck %s
target triple = "aarch64--linux-gnu"
@buf = global [16 x i8] c"\0A\0A\0A\0A\0A\0A\0A\0A\0A\0A\0A\0A\0A\0A\0A\0A", align 1

define i32 @test1(i32 %len, i8* nocapture readonly %buf) #0 {
entry:
  %cmp.13 = icmp sgt i32 %len, 1
  br i1 %cmp.13, label %while.body.lr.ph, label %while.end

while.body.lr.ph:                                 ; preds = %entry
  br label %while.body

while.body:
;CHECK-LABEL: while.body:
;CHECK-NEXT:    %indvar = phi i32 [ %indvar.next, %while.body ], [ 0, %while.body.lr.ph ]
;CHECK-NEXT:    %sum4.015 = phi i64 [ 0, %while.body.lr.ph ], [ %add, %while.body ]
;CHECK-NOT:     %sub5 = add nsw i32 %len.addr.014, -1
;CHECK-NOT:     %sub5 = add nsw i32 %len.addr.014, -2
;CHECK:    br i1 %exitcond, label %while.cond.while.end_crit_edge, label %while.body

  %sum4.015 = phi i64 [ 0, %while.body.lr.ph ], [ %add4, %while.body ]
  %len.addr.014 = phi i32 [ %len, %while.body.lr.ph ], [ %sub5, %while.body ]
  %idxprom = sext i32 %len.addr.014 to i64
  %arrayidx = getelementptr inbounds i8, i8* %buf, i64 %idxprom
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i64
  %add = add i64 %conv, %sum4.015
  %sub = add nsw i32 %len.addr.014, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds i8, i8* %buf, i64 %idxprom1
  %1 = load i8, i8* %arrayidx2, align 1
  %conv3 = zext i8 %1 to i64
  %add4 = add i64 %add, %conv3
  %sub5 = add nsw i32 %len.addr.014, -2
  %cmp = icmp sgt i32 %sub5, 1
  br i1 %cmp, label %while.body, label %while.cond.while.end_crit_edge

while.cond.while.end_crit_edge:                   ; preds = %while.body
  %add4.lcssa = phi i64 [ %add4, %while.body ]
  %phitmp = trunc i64 %add4.lcssa to i32
  br label %while.end

while.end:                                        ; preds = %while.cond.while.end_crit_edge, %entry
  %sum4.0.lcssa = phi i32 [ %phitmp, %while.cond.while.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum4.0.lcssa
  unreachable
}

