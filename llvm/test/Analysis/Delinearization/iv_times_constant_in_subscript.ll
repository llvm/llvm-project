; RUN: opt < %s -analyze -delinearize | FileCheck %s

; Derived from the following code:
;
; void foo(long n, long m, long b, double A[n][m]) {
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;       A[2i+b][2j] = 1.0;
; }

; AddRec: {{((%m * %b * 8) + %A),+,(2 * %m * 8)}<%for.i>,+,(2 * 8)}<%for.j>
; CHECK: Base offset: %A
; CHECK: ArrayDecl[UnknownSize][%m] with elements of 8 bytes.
; CHECK: ArrayRef[{%b,+,2}<nsw><%for.i>][{0,+,2}<nuw><%for.j>]


define void @foo(i64 %n, i64 %m, i64 %b, double* %A) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.inc ]
  %outerdim = mul nsw i64 %i, 2
  %outerdim2 = add nsw i64 %outerdim, %b
  %tmp = mul nsw i64 %outerdim2, %m
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %j.inc, %for.j ]
  %prodj = mul i64 %j, 2
  %vlaarrayidx.sum = add i64 %prodj, %tmp
  %arrayidx = getelementptr inbounds double, double* %A, i64 %vlaarrayidx.sum
  store double 1.0, double* %arrayidx
  %j.inc = add nsw i64 %j, 1
  %j.exitcond = icmp eq i64 %j.inc, %m
  br i1 %j.exitcond, label %for.i.inc, label %for.j

for.i.inc:
  %i.inc = add nsw i64 %i, 1
  %i.exitcond = icmp eq i64 %i.inc, %n
  br i1 %i.exitcond, label %end, label %for.i

end:
  ret void
}
