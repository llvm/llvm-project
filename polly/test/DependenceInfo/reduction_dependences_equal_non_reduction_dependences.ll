; RUN: opt %loadPolly -basic-aa -polly-stmt-granularity=bb -polly-print-dependences -disable-output < %s | FileCheck %s
;
; This loopnest contains a reduction which imposes the same dependences as the
; accesses to the array A. We need to ensure we keep the dependences of A.
;
; CHECK:      RAW dependences:
; CHECK-NEXT:     { Stmt_for_body[i0] -> Stmt_for_body[1 + i0] : 0 <= i0 <= 1022 }
; CHECK-NEXT: WAR dependences:
; CHECK-NEXT:     { Stmt_for_body[i0] -> Stmt_for_body[1 + i0] : 0 <= i0 <= 1022 }
; CHECK-NEXT: WAW dependences:
; CHECK-NEXT:     { Stmt_for_body[i0] -> Stmt_for_body[1 + i0] : 0 <= i0 <= 1022 }
; CHECK-NEXT: Reduction dependences:
; CHECK-NEXT:     { Stmt_for_body[i0] -> Stmt_for_body[1 + i0] : 0 <= i0 <= 1022 }
;
;
;    void AandSum(int *restrict sum, int *restrict A) {
;      for (int i = 0; i < 1024; i++) {
;        A[i] = A[i] + A[i - 1];
;        A[i - 1] = A[i] + A[i - 2];
;        *sum += i;
;      }
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @AandSum(i32* noalias %sum, i32* noalias %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.0
  %tmp = load i32, i32* %arrayidx, align 4
  %sub = add nsw i32 %i.0, -1
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i32 %sub
  %tmp1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %tmp, %tmp1
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.0
  store i32 %add, i32* %arrayidx2, align 4
  %sub4 = add nsw i32 %i.0, -2
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i32 %sub4
  %tmp2 = load i32, i32* %arrayidx5, align 4
  %add6 = add nsw i32 %add, %tmp2
  %sub7 = add nsw i32 %i.0, -1
  %arrayidx8 = getelementptr inbounds i32, i32* %A, i32 %sub7
  store i32 %add6, i32* %arrayidx8, align 4
  %tmp3 = load i32, i32* %sum, align 4
  %add9 = add nsw i32 %tmp3, %i.0
  store i32 %add9, i32* %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
