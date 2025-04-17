; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -polly-allow-nonaffine '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; void foo(long n, double A[], int INDEX[]) {
;   for (long i = 0; i < n; i++)
;     A[INDEX[i]] = i;
; }

define void @foo(i64 %n, ptr noalias %A, ptr noalias %INDEX) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %INDEX, i64 %i
  %val = load i64, ptr %arrayidx
  %arrayidx1 = getelementptr inbounds double, ptr %A, i64 %val
  store double 1.0, ptr %arrayidx1
  %inc = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK:      p0: %n
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_for_body[i0] : 0 <= i0 < n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_for_body[i0] -> MemRef_INDEX[i0] };
; CHECK-NEXT:         MayWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_for_body[i0] -> MemRef_A[o0] };
; CHECK-NEXT: }
