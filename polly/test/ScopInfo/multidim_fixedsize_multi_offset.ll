; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK:      Context:
; CHECK-NEXT: {  :  }
; CHECK:      Assumed Context:
; CHECK-NEXT: {  :  }
; CHECK:      Invalid Context:
; CHECK-NEXT: {  : false }
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_for_body[i0] : 0 <= i0 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0] -> MemRef_A[i0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0] -> MemRef_A[i0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0] -> MemRef_A[1 + i0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0] -> MemRef_A[1 + i0, 0] };
; CHECK-NEXT: }
;
;    void f(int A[][2]) {
;      int(*B)[2] = &A[0][0];
;      int(*C)[2] = &A[1][0];
;      for (int i = 0; i < 100; i++) {
;        B[i][0]++;
;        C[i][0]++;
;      }
;    }
;
; Verify that the additional offset to A by accessing it through C is taken into
; account.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
entry:
  %arrayidx3 = getelementptr inbounds [2 x i32], ptr %A, i64 1, i64 0
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx5 = getelementptr inbounds [2 x i32], ptr %A, i64 %indvars.iv, i64 0
  %tmp1 = load i32, ptr %arrayidx5, align 4
  %inc = add nsw i32 %tmp1, 1
  store i32 %inc, ptr %arrayidx5, align 4
  %arrayidx8 = getelementptr inbounds [2 x i32], ptr %arrayidx3, i64 %indvars.iv, i64 0
  %tmp2 = load i32, ptr %arrayidx8, align 4
  %inc9 = add nsw i32 %tmp2, 1
  store i32 %inc9, ptr %arrayidx8, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
