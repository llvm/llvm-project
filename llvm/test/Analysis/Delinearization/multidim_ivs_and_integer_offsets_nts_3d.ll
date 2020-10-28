; RUN: opt < %s -analyze -enable-new-pm=0 -delinearize | FileCheck %s
; RUN: opt < %s -passes='print<delinearization>' -disable-output 2>&1 | FileCheck %s

; void foo(long n, long m, long o, long p, double A[n][m][o+p]) {
;
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;       for (long k = 0; k < o; k++)
;         A[i+3][j-4][k+7] = 1.0;
; }

; AddRec: {{{(56 + (8 * (-4 + (3 * %m)) * (%o + %p)) + %A),+,(8 * (%o + %p) * %m)}<%for.cond4.preheader.lr.ph.us>,+,(8 * (%o + %p))}<%for.body6.lr.ph.us.us>,+,8}<%for.body6.us.us>
; CHECK: Base offset: %A
; CHECK: ArrayDecl[UnknownSize][%m][(%o + %p)] with elements of 8 bytes.
; CHECK: ArrayRef[{3,+,1}<nuw><%for.cond4.preheader.lr.ph.us>][{-4,+,1}<nw><%for.body6.lr.ph.us.us>][{7,+,1}<nw><%for.body6.us.us>]

define void @foo(i64 %n, i64 %m, i64 %o, i64 %p, double* nocapture %A) nounwind uwtable {
entry:
  %add = add nsw i64 %p, %o
  %cmp22 = icmp sgt i64 %n, 0
  br i1 %cmp22, label %for.cond1.preheader.lr.ph, label %for.end16

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp220 = icmp sgt i64 %m, 0
  %cmp518 = icmp sgt i64 %o, 0
  br i1 %cmp220, label %for.cond4.preheader.lr.ph.us, label %for.end16

for.inc14.us:                                     ; preds = %for.cond4.preheader.lr.ph.us, %for.inc11.us.us
  %inc15.us = add nsw i64 %i.023.us, 1
  %exitcond43 = icmp eq i64 %inc15.us, %n
  br i1 %exitcond43, label %for.end16, label %for.cond4.preheader.lr.ph.us

for.cond4.preheader.lr.ph.us:                     ; preds = %for.inc14.us, %for.cond1.preheader.lr.ph
  %i.023.us = phi i64 [ %inc15.us, %for.inc14.us ], [ 0, %for.cond1.preheader.lr.ph ]
  %add8.us = add nsw i64 %i.023.us, 3
  %0 = mul i64 %add8.us, %m
  %sub.us = add i64 %0, -4
  br i1 %cmp518, label %for.body6.lr.ph.us.us, label %for.inc14.us

for.inc11.us.us:                                  ; preds = %for.body6.us.us
  %inc12.us.us = add nsw i64 %j.021.us.us, 1
  %exitcond42 = icmp eq i64 %inc12.us.us, %m
  br i1 %exitcond42, label %for.inc14.us, label %for.body6.lr.ph.us.us

for.body6.lr.ph.us.us:                            ; preds = %for.cond4.preheader.lr.ph.us, %for.inc11.us.us
  %j.021.us.us = phi i64 [ %inc12.us.us, %for.inc11.us.us ], [ 0, %for.cond4.preheader.lr.ph.us ]
  %tmp.us.us = add i64 %sub.us, %j.021.us.us
  %tmp17.us.us = mul i64 %tmp.us.us, %add
  br label %for.body6.us.us

for.body6.us.us:                                  ; preds = %for.body6.us.us, %for.body6.lr.ph.us.us
  %k.019.us.us = phi i64 [ 0, %for.body6.lr.ph.us.us ], [ %inc.us.us, %for.body6.us.us ]
  %arrayidx.sum.us.us = add i64 %k.019.us.us, 7
  %arrayidx9.sum.us.us = add i64 %arrayidx.sum.us.us, %tmp17.us.us
  %arrayidx10.us.us = getelementptr inbounds double, double* %A, i64 %arrayidx9.sum.us.us
  store double 1.000000e+00, double* %arrayidx10.us.us, align 8
  %inc.us.us = add nsw i64 %k.019.us.us, 1
  %exitcond = icmp eq i64 %inc.us.us, %o
  br i1 %exitcond, label %for.inc11.us.us, label %for.body6.us.us

for.end16:                                        ; preds = %for.cond1.preheader.lr.ph, %for.inc14.us, %entry
  ret void
}
