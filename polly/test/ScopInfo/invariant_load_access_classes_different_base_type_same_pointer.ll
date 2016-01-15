; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s --check-prefix=CODEGEN
;
;    int U;
;    void f(int *A) {
;      for (int i = 0; i < 1000; i++)
;        A[i] = (*(int *)&U) + (int)(*(float *)&U);
;    }
;
; CHECK:      Invariant Accesses: {
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0] -> MemRef_U[0] };
; CHECK-NEXT:         Execution Context: {  :  }
; CHECK-NEXT: }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_for_body[i0] : 0 <= i0 <= 999 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
;
; CODEGEN: entry:
; CODEGEN:   %U.f.preload.s2a = alloca float
; CODEGEN:   br label %polly.split_new_and_old
;
; CODEGEN: polly.preload.begin:
; CODEGEN:   %U.load1 = load float, float* bitcast (i32* @U to float*)
; TODO FIXME There should not be a bitcast but either a real conversion or
;            another load as one type is FP the other is not.
; CODEGEN:   %0 = bitcast float %U.load1 to i32
; CODEGEN:   store float %U.load1, float* %U.f.preload.s2a
;
; CODEGEN:     polly.merge_new_and_old:
; CODEGEN-NOT:   merge = phi
;
; CODEGEN: polly.loop_exit:
; CODEGEN-NOT:   final_reload
;
; CODEGEN: polly.stmt.for.body:
; CODEGEN:   %p_conv = fptosi float %U.load1 to i32
; CODEGEN:   %p_add = add nsw i32 %0, %p_conv
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@U = common global i32 0, align 4

define void @f(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1000
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %U.i = load i32, i32* @U, align 4
  %U.cast = bitcast i32 *@U to float*
  %U.f = load float, float* %U.cast, align 4
  %conv = fptosi float %U.f to i32
  %add = add nsw i32 %U.i, %conv
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %add, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
