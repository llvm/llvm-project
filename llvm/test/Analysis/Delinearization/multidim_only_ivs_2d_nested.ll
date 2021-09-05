; RUN: opt < %s -passes='print<delinearization>' -disable-output 2>&1 | FileCheck %s
; XFAIL: *
; We do not recognize anymore variable size arrays.

; extern void bar(long n, long m, double A[n][m]);
;
; void foo(long a, long b) {
;   for (long n = 1; n < a; ++n)
;   for (long m = 1; m < b; ++m) {
;     double A[n][m];
;     for (long i = 0; i < n; i++)
;       for (long j = 0; j < m; j++)
;         A[i][j] = 1.0;
;     bar(n, m, A);
;   }
; }

; AddRec: {{%vla.us,+,{8,+,8}<%for.cond7.preheader.lr.ph.split.us.us>}<%for.body9.lr.ph.us.us>,+,8}<%for.body9.us.us>
; CHECK: Base offset: %vla.us
; CHECK: ArrayDecl[UnknownSize][{1,+,1}<%for.cond7.preheader.lr.ph.split.us.us>] with elements of sizeof(double) bytes.
; CHECK: ArrayRef[{0,+,1}<nuw><nsw><%for.body9.lr.ph.us.us>][{0,+,1}<nuw><nsw><%for.body9.us.us>]

define void @foo(i64 %a, i64 %b) nounwind uwtable {
entry:
  %cmp43 = icmp sgt i64 %a, 1
  br i1 %cmp43, label %for.cond1.preheader.lr.ph, label %for.end19

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp224 = icmp sgt i64 %b, 1
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc17, %for.cond1.preheader.lr.ph
  %indvars.iv51 = phi i64 [ 1, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next52, %for.inc17 ]
  br i1 %cmp224, label %for.cond7.preheader.lr.ph.split.us.us, label %for.inc17

for.end13.us:                                     ; preds = %for.inc11.us.us
  call void @bar(i64 %indvars.iv51, i64 %indvars.iv48, double* %vla.us) nounwind
  call void @llvm.stackrestore(i8* %1)
  %indvars.iv.next49 = add i64 %indvars.iv48, 1
  %exitcond54 = icmp eq i64 %indvars.iv.next49, %b
  br i1 %exitcond54, label %for.inc17, label %for.cond7.preheader.lr.ph.split.us.us

for.inc11.us.us:                                  ; preds = %for.body9.us.us
  %inc12.us.us = add nsw i64 %i.023.us.us, 1
  %exitcond53 = icmp eq i64 %inc12.us.us, %indvars.iv51
  br i1 %exitcond53, label %for.end13.us, label %for.body9.lr.ph.us.us

for.body9.lr.ph.us.us:                            ; preds = %for.cond7.preheader.lr.ph.split.us.us, %for.inc11.us.us
  %i.023.us.us = phi i64 [ 0, %for.cond7.preheader.lr.ph.split.us.us ], [ %inc12.us.us, %for.inc11.us.us ]
  %0 = mul nsw i64 %i.023.us.us, %indvars.iv48
  br label %for.body9.us.us

for.body9.us.us:                                  ; preds = %for.body9.us.us, %for.body9.lr.ph.us.us
  %j.021.us.us = phi i64 [ 0, %for.body9.lr.ph.us.us ], [ %inc.us.us, %for.body9.us.us ]
  %arrayidx.sum.us.us = add i64 %j.021.us.us, %0
  %arrayidx10.us.us = getelementptr inbounds double, double* %vla.us, i64 %arrayidx.sum.us.us
  store double 1.000000e+00, double* %arrayidx10.us.us, align 8
  %inc.us.us = add nsw i64 %j.021.us.us, 1
  %exitcond50 = icmp eq i64 %inc.us.us, %indvars.iv48
  br i1 %exitcond50, label %for.inc11.us.us, label %for.body9.us.us

for.cond7.preheader.lr.ph.split.us.us:            ; preds = %for.cond1.preheader, %for.end13.us
  %indvars.iv48 = phi i64 [ %indvars.iv.next49, %for.end13.us ], [ 1, %for.cond1.preheader ]
  %1 = call i8* @llvm.stacksave()
  %2 = mul nuw i64 %indvars.iv48, %indvars.iv51
  %vla.us = alloca double, i64 %2, align 16
  br label %for.body9.lr.ph.us.us

for.inc17:                                        ; preds = %for.end13.us, %for.cond1.preheader
  %indvars.iv.next52 = add i64 %indvars.iv51, 1
  %exitcond55 = icmp eq i64 %indvars.iv.next52, %a
  br i1 %exitcond55, label %for.end19, label %for.cond1.preheader

for.end19:                                        ; preds = %for.inc17, %entry
  ret void
}

declare i8* @llvm.stacksave() nounwind
declare void @bar(i64, i64, double*)
declare void @llvm.stackrestore(i8*) nounwind
