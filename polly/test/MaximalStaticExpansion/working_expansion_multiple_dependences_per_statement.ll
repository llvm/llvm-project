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
;   for (j = 0; j < Ni; j++) {
;     for (int i = 0; i<Nj; i++)
;       B[i] = i;
;
;     for (int i = 0; i<Nj; i++)
;       D[i] = i;
;
;     A[j] = B[j];
;     C[j] = D[j];
;   }
; }
;
; Check that expanded SAI are created
;
; CHECK: double MemRef_B_Stmt_for_body4_expanded[10000][10000]; // Element size 8
; CHECK: double MemRef_D_Stmt_for_body9_expanded[10000][10000]; // Element size 8
; CHECK: i64 MemRef_A_Stmt_for_end15_expanded[10000]; // Element size 8
; CHECK: i64 MemRef_C_Stmt_for_end15_expanded[10000]; // Element size 8
;
; Check that the memory accesses are modified
; CHECK: new: { Stmt_for_body4[i0, i1] -> MemRef_B_Stmt_for_body4_expanded[i0, i1] };
; CHECK: new: { Stmt_for_body9[i0, i1] -> MemRef_D_Stmt_for_body9_expanded[i0, i1] };
; CHECK: new: { Stmt_for_end15[i0] -> MemRef_B_Stmt_for_body4_expanded[i0, i0] };
; CHECK: new: { Stmt_for_end15[i0] -> MemRef_A_Stmt_for_end15_expanded[i0] };
; CHECK: new: { Stmt_for_end15[i0] -> MemRef_D_Stmt_for_body9_expanded[i0, i0] };
; CHECK: new: { Stmt_for_end15[i0] -> MemRef_C_Stmt_for_end15_expanded[i0] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @mse(ptr %A, ptr %B, ptr %C, ptr %D) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.end15
  %indvars.iv7 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next8, %for.end15 ]
  br label %for.body4

for.body4:                                        ; preds = %for.body, %for.body4
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body4 ]
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx = getelementptr inbounds double, ptr %B, i64 %indvars.iv
  store double %conv, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 10000
  br i1 %exitcond, label %for.body4, label %for.end

for.end:                                          ; preds = %for.body4
  br label %for.body9

for.body9:                                        ; preds = %for.end, %for.body9
  %indvars.iv4 = phi i64 [ 0, %for.end ], [ %indvars.iv.next5, %for.body9 ]
  %1 = trunc i64 %indvars.iv4 to i32
  %conv10 = sitofp i32 %1 to double
  %arrayidx12 = getelementptr inbounds double, ptr %D, i64 %indvars.iv4
  store double %conv10, ptr %arrayidx12, align 8
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1
  %exitcond6 = icmp ne i64 %indvars.iv.next5, 10000
  br i1 %exitcond6, label %for.body9, label %for.end15

for.end15:                                        ; preds = %for.body9
  %arrayidx17 = getelementptr inbounds double, ptr %B, i64 %indvars.iv7
  %2 = load i64, ptr %arrayidx17, align 8
  %arrayidx19 = getelementptr inbounds double, ptr %A, i64 %indvars.iv7
  store i64 %2, ptr %arrayidx19, align 8
  %arrayidx21 = getelementptr inbounds double, ptr %D, i64 %indvars.iv7
  %3 = load i64, ptr %arrayidx21, align 8
  %arrayidx23 = getelementptr inbounds double, ptr %C, i64 %indvars.iv7
  store i64 %3, ptr %arrayidx23, align 8
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1
  %exitcond9 = icmp ne i64 %indvars.iv.next8, 10000
  br i1 %exitcond9, label %for.body, label %for.end26

for.end26:                                        ; preds = %for.end15
  ret void
}
