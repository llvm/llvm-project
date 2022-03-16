; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(i64* nocapture %a) nounwind {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.j ]
  %i.inc = add nsw i64 %i, 1
  %exitcond.i = icmp sge i64 %i.inc, 2048
  br i1 %exitcond.i, label %return, label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %j.inc, %body ]
  %j.inc = add nsw i64 %j, 1
  %exitcond.j = icmp slt i64 %j.inc, 1024
  br i1 %exitcond.j, label %body, label %for.i

body:
  %scevgep = getelementptr i64, i64* %a, i64 %j
  store i64 %j, i64* %scevgep
  br label %for.j

return:
  ret void
}


; CHECK:      Statements {
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_body[i0, i1] : 0 <= i0 <= 2046 and 0 <= i1 <= 1022 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_body[i0, i1] -> [i0, i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_body[i0, i1] -> MemRef_a[i1] };
; CHECK-NEXT: }
