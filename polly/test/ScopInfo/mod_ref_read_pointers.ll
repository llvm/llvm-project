; RUN: opt %loadPolly -basicaa -polly-scops -analyze -polly-allow-modref-calls \
; RUN: < %s | FileCheck %s
; RUN: opt %loadPolly -basicaa -polly-codegen -disable-output \
; RUN: -polly-allow-modref-calls < %s
;
; Check that the call to func will "read" not only the A array but also the
; B array. The reason is the readonly annotation of func.
;
; CHECK:      Stmt_for_body
; CHECK-NEXT:  Domain :=
; CHECK-NEXT:      { Stmt_for_body[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:  Schedule :=
; CHECK-NEXT:      { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:  ReadAccess :=  [Reduction Type: NONE]
; CHECK-NEXT:      { Stmt_for_body[i0] -> MemRef_B[i0] };
; CHECK-NEXT:  MustWriteAccess :=  [Reduction Type: NONE]
; CHECK-NEXT:      { Stmt_for_body[i0] -> MemRef_A[2 + i0] };
; CHECK-DAG:   ReadAccess :=  [Reduction Type: NONE]
; CHECK-DAG:       { Stmt_for_body[i0] -> MemRef_B[o0] };
; CHECK-DAG:   ReadAccess :=  [Reduction Type: NONE]
; CHECK-DAG:       { Stmt_for_body[i0] -> MemRef_A[o0] };
;
;    #pragma readonly
;    int func(int *A);
;
;    void jd(int *restrict A, int *restrict B) {
;      for (int i = 0; i < 1024; i++)
;        A[i + 2] = func(A) + B[i];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* noalias %A, i32* noalias %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call = call i32 @func(i32* %A)
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %call, %tmp
  %tmp2 = add nsw i64 %indvars.iv, 2
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 %tmp2
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare i32 @func(i32*) #1

attributes #1 = { nounwind readonly }
