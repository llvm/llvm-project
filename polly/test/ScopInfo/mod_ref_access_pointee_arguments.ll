; RUN: opt %loadPolly -basicaa -polly-scops -analyze -polly-allow-modref-calls \
; RUN:  < %s | FileCheck %s
; RUN: opt %loadPolly -basicaa -polly-codegen -polly-allow-modref-calls \
; RUN: -disable-output < %s
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

define void @jd(i32* noalias %A, i32* noalias %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %bc = bitcast i32* %arrayidx to i8*
  call void @f(i8* %bc, i32 1, i32 1, i32 1)
  %tmp = load i32, i32* %arrayidx2
  store i32 %tmp, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @f(i8*, i32, i32, i32) #0

attributes #0 = { argmemonly nounwind }
