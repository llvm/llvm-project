; RUN: opt -S -passes=loop-reroll   %s | FileCheck %s
declare i32 @goo(i32, i32)

@buf = external global i8*
@aaa = global [16 x i8] c"\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F\10", align 1

define i32 @test1(i32 %len) {
entry:
  br label %while.body

while.body:
;CHECK-LABEL: while.body:
;CHECK-NEXT:    %indvar = phi i64 [ %indvar.next, %while.body ], [ 0, %entry ]
;CHECK-NEXT:    %sum44.020 = phi i64 [ 0, %entry ], [ %add, %while.body ]
;CHECK-NEXT:    %0 = trunc i64 %indvar to i32
;CHECK-NEXT:    %scevgep = getelementptr [16 x i8], [16 x i8]* @aaa, i64 0, i64 %indvar
;CHECK-NEXT:    [[T2:%[0-9]+]] = load i8, i8* %scevgep, align 1
;CHECK-NEXT:    %conv = zext i8 [[T2]] to i64
;CHECK-NEXT:    %add = add i64 %conv, %sum44.020
;CHECK-NEXT:    %indvar.next = add i64 %indvar, 1
;CHECK-NEXT:    %exitcond = icmp eq i32 %0, 15
;CHECK-NEXT:    br i1 %exitcond, label %while.end, label %while.body

  %dec22 = phi i32 [ 4, %entry ], [ %dec, %while.body ]
  %buf.021 = phi i8* [ getelementptr inbounds ([16 x i8], [16 x i8]* @aaa, i64 0, i64 0), %entry ], [ %add.ptr, %while.body ]
  %sum44.020 = phi i64 [ 0, %entry ], [ %add9, %while.body ]
  %0 = load i8, i8* %buf.021, align 1
  %conv = zext i8 %0 to i64
  %add = add i64 %conv, %sum44.020
  %arrayidx1 = getelementptr inbounds i8, i8* %buf.021, i64 1
  %1 = load i8, i8* %arrayidx1, align 1
  %conv2 = zext i8 %1 to i64
  %add3 = add i64 %add, %conv2
  %arrayidx4 = getelementptr inbounds i8, i8* %buf.021, i64 2
  %2 = load i8, i8* %arrayidx4, align 1
  %conv5 = zext i8 %2 to i64
  %add6 = add i64 %add3, %conv5
  %arrayidx7 = getelementptr inbounds i8, i8* %buf.021, i64 3
  %3 = load i8, i8* %arrayidx7, align 1
  %conv8 = zext i8 %3 to i64
  %add9 = add i64 %add6, %conv8
  %add.ptr = getelementptr inbounds i8, i8* %buf.021, i64 4
  %dec = add nsw i32 %dec22, -1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body
  %conv11 = trunc i64 %add9 to i32
  %call = tail call i32 @goo(i32 0, i32 %conv11)
  unreachable
}

define i32 @test2(i32 %N, i32* nocapture readonly %a, i32 %S) {
entry:
  %cmp.9 = icmp sgt i32 %N, 0
  br i1 %cmp.9, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:
  br label %for.cond.cleanup

for.cond.cleanup:
  %S.addr.0.lcssa = phi i32 [ %add2, %for.cond.for.cond.cleanup_crit_edge ], [ %S, %entry ]
  ret i32 %S.addr.0.lcssa

for.body:
;CHECK-LABEL: for.body:
;CHECK-NEXT:    %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %for.body.lr.ph ]
;CHECK-NEXT:    %S.addr.011 = phi i32 [ %S, %for.body.lr.ph ], [ %add, %for.body ]
;CHECK-NEXT:    %4 = trunc i64 %indvar to i32
;CHECK-NEXT:    %scevgep = getelementptr i32, i32* %a, i64 %indvar
;CHECK-NEXT:    %5 = load i32, i32* %scevgep, align 4
;CHECK-NEXT:    %add = add nsw i32 %5, %S.addr.011
;CHECK-NEXT:    %indvar.next = add i64 %indvar, 1
;CHECK-NEXT:    %exitcond = icmp eq i32 %4, %3
;CHECK-NEXT:    br i1 %exitcond, label %for.cond.for.cond.cleanup_crit_edge, label %for.body

  %i.012 = phi i32 [ 0, %for.body.lr.ph ], [ %add3, %for.body ]
  %S.addr.011 = phi i32 [ %S, %for.body.lr.ph ], [ %add2, %for.body ]
  %a.addr.010 = phi i32* [ %a, %for.body.lr.ph ], [ %incdec.ptr1, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %a.addr.010, i64 1
  %0 = load i32, i32* %a.addr.010, align 4
  %add = add nsw i32 %0, %S.addr.011
  %incdec.ptr1 = getelementptr inbounds i32, i32* %a.addr.010, i64 2
  %1 = load i32, i32* %incdec.ptr, align 4
  %add2 = add nsw i32 %add, %1
  %add3 = add nsw i32 %i.012, 2
  %cmp = icmp slt i32 %add3, %N
  br i1 %cmp, label %for.body, label %for.cond.for.cond.cleanup_crit_edge
}

define i32 @test3(i32* nocapture readonly %buf, i32 %len) #0 {
entry:
  %cmp10 = icmp sgt i32 %len, 1
  br i1 %cmp10, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
;CHECK-LABEL: while.body:
;CHECK-NEXT:  %indvar = phi i64 [ %indvar.next, %while.body ], [ 0, %while.body.preheader ]
;CHECK-NEXT:  %S.012 = phi i32 [ %add, %while.body ], [ undef, %while.body.preheader ]
;CHECK-NEXT:  %4 = trunc i64 %indvar to i32
;CHECK-NEXT:  %5 = mul nsw i64 %indvar, -1
;CHECK-NEXT:  %scevgep = getelementptr i32, i32* %buf, i64 %5
;CHECK-NEXT:  %6 = load i32, i32* %scevgep, align 4
;CHECK-NEXT:  %add = add nsw i32 %6, %S.012
;CHECK-NEXT:  %indvar.next = add i64 %indvar, 1
;CHECK-NEXT:  %exitcond = icmp eq i32 %4, %3
;CHECK-NEXT:  br i1 %exitcond, label %while.end.loopexit, label %while.body

  %i.013 = phi i32 [ %sub, %while.body ], [ %len, %while.body.preheader ]
  %S.012 = phi i32 [ %add2, %while.body ], [ undef, %while.body.preheader ]
  %buf.addr.011 = phi i32* [ %add.ptr, %while.body ], [ %buf, %while.body.preheader ]
  %0 = load i32, i32* %buf.addr.011, align 4
  %add = add nsw i32 %0, %S.012
  %arrayidx1 = getelementptr inbounds i32, i32* %buf.addr.011, i64 -1
  %1 = load i32, i32* %arrayidx1, align 4
  %add2 = add nsw i32 %add, %1
  %add.ptr = getelementptr inbounds i32, i32* %buf.addr.011, i64 -2
  %sub = add nsw i32 %i.013, -2
  %cmp = icmp sgt i32 %sub, 1
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %S.0.lcssa = phi i32 [ undef, %entry ], [ %add2, %while.end.loopexit ]
  ret i32 %S.0.lcssa
}

