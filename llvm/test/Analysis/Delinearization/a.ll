; RUN: opt < %s -analyze -enable-new-pm=0 -delinearize | FileCheck %s
; RUN: opt < %s -passes='print<delinearization>' -disable-output 2>&1 | FileCheck %s
;
; void foo(long n, long m, long o, int A[n][m][o]) {
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;       for (long k = 0; k < o; k++)
;         A[2*i+3][3*j-4][5*k+7] = 1;
; }

; AddRec: {{{(28 + (4 * (-4 + (3 * %m)) * %o) + %A),+,(8 * %m * %o)}<%for.i>,+,(12 * %o)}<%for.j>,+,20}<%for.k>
; CHECK: Base offset: %A
; CHECK: ArrayDecl[UnknownSize][%m][%o] with elements of 4 bytes.
; CHECK: ArrayRef[{3,+,2}<nw><%for.i>][{-4,+,3}<nw><%for.j>][{7,+,5}<nw><%for.k>]

define void @foo(i64 %n, i64 %m, i64 %o, i32* nocapture %A) #0 {
entry:
  %cmp32 = icmp sgt i64 %n, 0
  br i1 %cmp32, label %for.cond1.preheader.lr.ph, label %for.end17

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp230 = icmp sgt i64 %m, 0
  %cmp528 = icmp sgt i64 %o, 0
  br i1 %cmp230, label %for.i, label %for.end17

for.inc15.us:                                     ; preds = %for.inc12.us.us, %for.i
  %inc16.us = add nsw i64 %i.033.us, 1
  %exitcond55 = icmp eq i64 %inc16.us, %n
  br i1 %exitcond55, label %for.end17, label %for.i

for.i:                     ; preds = %for.cond1.preheader.lr.ph, %for.inc15.us
  %i.033.us = phi i64 [ %inc16.us, %for.inc15.us ], [ 0, %for.cond1.preheader.lr.ph ]
  %mul8.us = shl i64 %i.033.us, 1
  %add9.us = add nsw i64 %mul8.us, 3
  %0 = mul i64 %add9.us, %m
  %sub.us = add i64 %0, -4
  br i1 %cmp528, label %for.j, label %for.inc15.us

for.inc12.us.us:                                  ; preds = %for.k
  %inc13.us.us = add nsw i64 %j.031.us.us, 1
  %exitcond54 = icmp eq i64 %inc13.us.us, %m
  br i1 %exitcond54, label %for.inc15.us, label %for.j

for.j:                            ; preds = %for.i, %for.inc12.us.us
  %j.031.us.us = phi i64 [ %inc13.us.us, %for.inc12.us.us ], [ 0, %for.i ]
  %mul7.us.us = mul nsw i64 %j.031.us.us, 3
  %tmp.us.us = add i64 %sub.us, %mul7.us.us
  %tmp27.us.us = mul i64 %tmp.us.us, %o
  br label %for.k

for.k:                                  ; preds = %for.k, %for.j
  %k.029.us.us = phi i64 [ 0, %for.j ], [ %inc.us.us, %for.k ]
  %mul.us.us = mul nsw i64 %k.029.us.us, 5
  %arrayidx.sum.us.us = add i64 %mul.us.us, 7
  %arrayidx10.sum.us.us = add i64 %arrayidx.sum.us.us, %tmp27.us.us
  %arrayidx11.us.us = getelementptr inbounds i32, i32* %A, i64 %arrayidx10.sum.us.us
  store i32 1, i32* %arrayidx11.us.us, align 4
  %inc.us.us = add nsw i64 %k.029.us.us, 1
  %exitcond = icmp eq i64 %inc.us.us, %o
  br i1 %exitcond, label %for.inc12.us.us, label %for.k

for.end17:                                        ; preds = %for.inc15.us, %for.cond1.preheader.lr.ph, %entry
  ret void
}
