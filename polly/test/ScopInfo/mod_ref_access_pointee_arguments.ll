; RUN: opt %loadPolly -basic-aa -polly-stmt-granularity=bb -polly-print-scops -polly-allow-modref-calls \
; RUN:     -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -basic-aa -polly-stmt-granularity=bb -polly-codegen -polly-allow-modref-calls \
; RUN:     -disable-output < %s
;
; Verify that we model the may-write access of the prefetch intrinsic
; correctly, thus that A is accessed by it but B is not.
;
; CHECK:      Stmt_for_body
; CHECK-NEXT:   Domain :=
; CHECK-NEXT:       { Stmt_for_body[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:   Schedule :=
; CHECK-NEXT:       { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:   MayWriteAccess := [Reduction Type: NONE]
; CHECK-NEXT:       { Stmt_for_body[i0] -> MemRef_A[o0] };
; CHECK-NEXT:   ReadAccess := [Reduction Type: NONE]
; CHECK-NEXT:       { Stmt_for_body[i0] -> MemRef_B[i0] };
; CHECK-NEXT:   MustWriteAccess :=  [Reduction Type: NONE]
; CHECK-NEXT:       { Stmt_for_body[i0] -> MemRef_A[i0] };
;
;    void jd(int *restirct A, int *restrict B) {
;      for (int i = 0; i < 1024; i++) {
;        @llvm.prefetch(A);
;        A[i] = B[i];
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(ptr noalias %A, ptr noalias %B) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %i
  %arrayidx1 = getelementptr inbounds i32, ptr %B, i64 %i
  call void @f(ptr %arrayidx, i32 1, i32 1, i32 1)
  %tmp = load i32, ptr %arrayidx1
  store i32 %tmp, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp ne i64 %i.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

declare void @f(ptr, i32, i32, i32) #0

attributes #0 = { argmemonly nounwind }
