; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Derived from the following code:
;
; void foo(long n, long m, double *A) {
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;       *(A + i * n + j) = 1.0;
;       *(A + j * m + i) = 2.0;
; }

; CHECK-NOT: Valid Region for Scop

define void @foo(i64 %n, i64 %m, ptr %A) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.inc ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %j.inc, %for.j ]
  %tmp = mul nsw i64 %i, %m
  %vlaarrayidx.sum = add i64 %j, %tmp
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %vlaarrayidx.sum
  store double 1.0, ptr %arrayidx
  %tmp1 = mul nsw i64 %j, %n
  %vlaarrayidx.sum1 = add i64 %i, %tmp1
  %arrayidx1 = getelementptr inbounds double, ptr %A, i64 %vlaarrayidx.sum1
  store double 1.0, ptr %arrayidx1
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
