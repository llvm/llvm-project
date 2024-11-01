; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-mse -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb "-passes=scop(print<polly-mse>)" -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-mse -polly-print-scops -pass-remarks-analysis="polly-mse" -disable-output < %s 2>&1| FileCheck %s --check-prefix=MSE
; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb "-passes=scop(print<polly-mse>)" -pass-remarks-analysis="polly-mse" -disable-output < %s 2>&1 | FileCheck %s --check-prefix=MSE
;
; Verify that the expansion of an array with load after store in a same statement is not done.
;
; Original source code :
;
; #define Ni 2000
; #define Nj 3000
;
; void mse(double A[Ni], double B[Nj], double C[Nj], double D[Nj]) {
;   int i,j;
;   for (i = 0; i < Ni; i++) {
;     for (int j = 0; j<Nj; j++) {
;       B[j] = j;
;       C[j] = B[j];
;     }
;   }
; }
;
; Check that C is expanded
;
; CHECK: i64 MemRef_C_Stmt_for_body4_expanded[10000][10000]; // Element size 8
; CHECK: new: { Stmt_for_body4[i0, i1] -> MemRef_C_Stmt_for_body4_expanded[i0, i1] };
;
; Check that B is not expanded
;
; CHECK-NOT: double MemRef_B_Stmt_for_body4_expanded[10000][10000]; // Element size 8
; MSE: MemRef_B has read after write to the same element in same statement. The dependences found during analysis may be wrong because Polly is not able to handle such case for now.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @mse(ptr %A, ptr %B, ptr %C, ptr %D) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.inc9
  %i.02 = phi i32 [ 0, %entry.split ], [ %inc10, %for.inc9 ]
  br label %for.body4

for.body4:                                        ; preds = %for.body, %for.body4
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body4 ]
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx = getelementptr inbounds double, ptr %B, i64 %indvars.iv
  store double %conv, ptr %arrayidx, align 8
  %arrayidx6 = getelementptr inbounds double, ptr %B, i64 %indvars.iv
  %1 = load i64, ptr %arrayidx6, align 8
  %arrayidx8 = getelementptr inbounds double, ptr %C, i64 %indvars.iv
  store i64 %1, ptr %arrayidx8, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 10000
  br i1 %exitcond, label %for.body4, label %for.inc9

for.inc9:                                         ; preds = %for.body4
  %inc10 = add nuw nsw i32 %i.02, 1
  %exitcond3 = icmp ne i32 %inc10, 10000
  br i1 %exitcond3, label %for.body, label %for.end11

for.end11:                                        ; preds = %for.inc9
  ret void
}
