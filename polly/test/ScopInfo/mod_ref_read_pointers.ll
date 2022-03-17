; RUN: opt %loadPolly -basic-aa -polly-print-scops -polly-allow-modref-calls \
; RUN:     -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -basic-aa -polly-codegen -disable-output \
; RUN:     -polly-allow-modref-calls < %s
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %call = call i32 @func(i32* %A)
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %i
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %call, %tmp
  %tmp1 = add nsw i64 %i, 2
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %tmp1
  store i32 %add, i32* %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp ne i64 %i.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

declare i32 @func(i32*) #0

attributes #0 = { nounwind readonly }
