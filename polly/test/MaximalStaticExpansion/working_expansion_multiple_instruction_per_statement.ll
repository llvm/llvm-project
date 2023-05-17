; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-mse -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb "-passes=scop(print<polly-mse>)" -disable-output < %s | FileCheck %s
;
; Verify that the accesses are correctly expanded
;
; Original source code :
;
; #define Ni 2000
; #define Nj 3000
;
; void mse(double A[Ni], double B[Nj], double C[Nj], double D[Nj]) {
;   int i,j;
;   for (j = 0; j < Nj; j++) {
;     for (int i = 0; i<Ni; i++) {
;       B[i] = i;
;       D[i] = i;
;     }
;     A[j] = B[j];
;     C[j] = D[j];
;   }
; }
;
; Check that expanded SAI are created
; CHECK: double MemRef_B_Stmt_for_body4_expanded[10000][10000]; // Element size 8
; CHECK: double MemRef_D_Stmt_for_body4_expanded[10000][10000]; // Element size 8
; CHECK: i64 MemRef_A_Stmt_for_end_expanded[10000]; // Element size 8
; CHECK: i64 MemRef_C_Stmt_for_end_expanded[10000]; // Element size 8
;
; Check that the memory access are modified
;
; CHECK: new: { Stmt_for_body4[i0, i1] -> MemRef_B_Stmt_for_body4_expanded[i0, i1] };
; CHECK: new: { Stmt_for_body4[i0, i1] -> MemRef_D_Stmt_for_body4_expanded[i0, i1] };
; CHECK: new: { Stmt_for_end[i0] -> MemRef_B_Stmt_for_body4_expanded[i0, i0] };
; CHECK: new: { Stmt_for_end[i0] -> MemRef_A_Stmt_for_end_expanded[i0] };
; CHECK: new: { Stmt_for_end[i0] -> MemRef_D_Stmt_for_body4_expanded[i0, i0] };
; CHECK: new: { Stmt_for_end[i0] -> MemRef_C_Stmt_for_end_expanded[i0] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @mse(ptr %A, ptr %B, ptr %C, ptr %D) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.end
  %indvars.iv3 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next4, %for.end ]
  br label %for.body4

for.body4:                                        ; preds = %for.body, %for.body4
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body4 ]
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx = getelementptr inbounds double, ptr %B, i64 %indvars.iv
  store double %conv, ptr %arrayidx, align 8
  %1 = trunc i64 %indvars.iv to i32
  %conv5 = sitofp i32 %1 to double
  %arrayidx7 = getelementptr inbounds double, ptr %D, i64 %indvars.iv
  store double %conv5, ptr %arrayidx7, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 10000
  br i1 %exitcond, label %for.body4, label %for.end

for.end:                                          ; preds = %for.body4
  %arrayidx9 = getelementptr inbounds double, ptr %B, i64 %indvars.iv3
  %2 = load i64, ptr %arrayidx9, align 8
  %arrayidx11 = getelementptr inbounds double, ptr %A, i64 %indvars.iv3
  store i64 %2, ptr %arrayidx11, align 8
  %arrayidx13 = getelementptr inbounds double, ptr %D, i64 %indvars.iv3
  %3 = load i64, ptr %arrayidx13, align 8
  %arrayidx15 = getelementptr inbounds double, ptr %C, i64 %indvars.iv3
  store i64 %3, ptr %arrayidx15, align 8
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  %exitcond5 = icmp ne i64 %indvars.iv.next4, 10000
  br i1 %exitcond5, label %for.body, label %for.end18

for.end18:                                        ; preds = %for.end
  ret void
}
