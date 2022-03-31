; RUN: opt %loadPolly -polly-invariant-load-hoisting=true -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void f(int *A, int *B, int *C) {
;      for (int i = 0; i < 1000; i++)
;        if (A[i] == *B)
;          A[i] = *C;
;    }
;
; Check that only the access to *B is hoisted but not the one to *C.
;
; CHECK: Invariant Accesses: {
; CHECK:     ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:         { Stmt_for_body__TO__if_end[i0] -> MemRef_B[0] };
; CHECK:     Execution Context: {  :  }
; CHECK: }
;
; CHECK: Statements {
; CHECK:   Stmt_for_body__TO__if_end
; CHECK:     ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:         { Stmt_for_body__TO__if_end[i0] -> MemRef_C[0] };
; CHECK: }

;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32* %B, i32* %C) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1000
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  %tmp1 = load i32, i32* %B, align 4
  %cmp1 = icmp eq i32 %tmp, %tmp1
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %tmp2 = load i32, i32* %C, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %tmp2, i32* %arrayidx3, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
