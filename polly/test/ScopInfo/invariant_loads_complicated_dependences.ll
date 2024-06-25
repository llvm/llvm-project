; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-invariant-load-hoisting=true -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK:      Invariant Accesses: {
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [LB, UB] -> { Stmt_for_body[i0] -> MemRef_LBptr[0] };
; CHECK-NEXT:         Execution Context: [LB, UB] -> {  :  }
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [LB, UB] -> { Stmt_do_cond[i0, i1] -> MemRef_UBptr[0] };
; CHECK-NEXT:         Execution Context: [LB, UB] -> {  :  }
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [LB, UB] -> { Stmt_if_then[i0, i1] -> MemRef_V[0] };
; CHECK-NEXT:         Execution Context: [LB, UB] -> {  : LB >= 6 or (UB > LB and UB >= 6) }
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [LB, UB] -> { Stmt_if_else[i0, i1] -> MemRef_U[0] };
; CHECK-NEXT:         Execution Context: [LB, UB] -> {  : LB <= 5 }
; CHECK-NEXT: }
;
;    void f(int *restrict A, int *restrict V, int *restrict U, int *restrict UB,
;           int *restrict LB) {
;      for (int i = 0; i < 100; i++) {
;        int j = /* invariant load */ *LB;
;        do {
;          if (j > 5)
;            A[i] += /* invariant load */ *V;
;          else
;            A[i] += /* invariant load */ *U;
;        } while (j++ < /* invariant load */ *UB);
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr noalias %A, ptr noalias %V, ptr noalias %U, ptr noalias %UBptr, ptr noalias %LBptr) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %LB = load i32, ptr %LBptr, align 4
  br label %do.body

do.body:                                          ; preds = %do.cond, %for.body
  %j.0 = phi i32 [ %LB, %for.body ], [ %inc, %do.cond ]
  %cmp1 = icmp sgt i32 %j.0, 5
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %do.body
  %tmp1 = load i32, ptr %V, align 4
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp2 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp2, %tmp1
  store i32 %add, ptr %arrayidx, align 4
  br label %if.end

if.else:                                          ; preds = %do.body
  %tmp3 = load i32, ptr %U, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp4 = load i32, ptr %arrayidx3, align 4
  %add4 = add nsw i32 %tmp4, %tmp3
  store i32 %add4, ptr %arrayidx3, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %do.cond

do.cond:                                          ; preds = %if.end
  %inc = add nsw i32 %j.0, 1
  %UB = load i32, ptr %UBptr, align 4
  %cmp5 = icmp slt i32 %j.0, %UB
  br i1 %cmp5, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  br label %for.inc

for.inc:                                          ; preds = %do.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
