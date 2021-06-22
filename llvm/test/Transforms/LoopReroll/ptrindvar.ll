; RUN: opt -S  -loop-reroll   %s | FileCheck %s
target triple = "aarch64--linux-gnu"

define i32 @test(i32* readonly %buf, i32* readnone %end) #0 {
entry:
  %cmp.9 = icmp eq i32* %buf, %end
  br i1 %cmp.9, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
;CHECK-LABEL: while.body:
;CHECK-NEXT:    %indvar = phi i64 [ %indvar.next, %while.body ], [ 0, %while.body.preheader ]
;CHECK-NEXT:    %S.011 = phi i32 [ %add, %while.body ], [ undef, %while.body.preheader ]
;CHECK-NEXT:    %scevgep = getelementptr i32, i32* %buf, i64 %indvar
;CHECK-NEXT:    %5 = load i32, i32* %scevgep, align 4
;CHECK-NEXT:    %add = add nsw i32 %5, %S.011
;CHECK-NEXT:    %indvar.next = add i64 %indvar, 1
;CHECK-NEXT:    %exitcond = icmp eq i64 %indvar, %4
;CHECK-NEXT:    br i1 %exitcond, label %while.end.loopexit, label %while.body

  %S.011 = phi i32 [ %add2, %while.body ], [ undef, %while.body.preheader ]
  %buf.addr.010 = phi i32* [ %add.ptr, %while.body ], [ %buf, %while.body.preheader ]
  %0 = load i32, i32* %buf.addr.010, align 4
  %add = add nsw i32 %0, %S.011
  %arrayidx1 = getelementptr inbounds i32, i32* %buf.addr.010, i64 1
  %1 = load i32, i32* %arrayidx1, align 4
  %add2 = add nsw i32 %add, %1
  %add.ptr = getelementptr inbounds i32, i32* %buf.addr.010, i64 2
  %cmp = icmp eq i32* %add.ptr, %end
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:
  %add2.lcssa = phi i32 [ %add2, %while.body ]
  br label %while.end

while.end:
  %S.0.lcssa = phi i32 [ undef, %entry ], [ %add2.lcssa, %while.end.loopexit ]
  ret i32 %S.0.lcssa
}

define i32 @test2(i32* readonly %buf, i32* readnone %end) #0 {
entry:
  %cmp.9 = icmp eq i32* %buf, %end
  br i1 %cmp.9, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
;CHECK-LABEL: while.body:
;CHECK-NEXT:    %indvar = phi i64 [ %indvar.next, %while.body ], [ 0, %while.body.preheader ]
;CHECK-NEXT:    %S.011 = phi i32 [ %add, %while.body ], [ undef, %while.body.preheader ]
;CHECK-NEXT:    %5 = mul nsw i64 %indvar, -1
;CHECK-NEXT:    %scevgep = getelementptr i32, i32* %buf, i64 %5
;CHECK-NEXT:    %6 = load i32, i32* %scevgep, align 4
;CHECK-NEXT:    %add = add nsw i32 %6, %S.011
;CHECK-NEXT:    %indvar.next = add i64 %indvar, 1
;CHECK-NEXT:    %exitcond = icmp eq i64 %indvar, %4
;CHECK-NEXT:    br i1 %exitcond, label %while.end.loopexit, label %while.body

  %S.011 = phi i32 [ %add2, %while.body ], [ undef, %while.body.preheader ]
  %buf.addr.010 = phi i32* [ %add.ptr, %while.body ], [ %buf, %while.body.preheader ]
  %0 = load i32, i32* %buf.addr.010, align 4
  %add = add nsw i32 %0, %S.011
  %arrayidx1 = getelementptr inbounds i32, i32* %buf.addr.010, i64 -1
  %1 = load i32, i32* %arrayidx1, align 4
  %add2 = add nsw i32 %add, %1
  %add.ptr = getelementptr inbounds i32, i32* %buf.addr.010, i64 -2
  %cmp = icmp eq i32* %add.ptr, %end
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:
  %add2.lcssa = phi i32 [ %add2, %while.body ]
  br label %while.end

while.end:
  %S.0.lcssa = phi i32 [ undef, %entry ], [ %add2.lcssa, %while.end.loopexit ]
  ret i32 %S.0.lcssa
}
