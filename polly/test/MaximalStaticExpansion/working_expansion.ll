; RUN: opt %loadPolly -polly-mse -analyze < %s | FileCheck %s
;
; Verify that the accesses are correctly expanded
;
; Original source code :
;
; #define Ni 2000
; #define Nj 3000
; 
; double mse(double A[Ni], double B[Nj]) {
;   int i;
;   double tmp = 6;
;   for (i = 0; i < Ni; i++) {
;     for (int j = 0; j<Nj; j++) {
;       B[j] = j;
;     }
;     A[i] = B[i]; 
;   }
;   return tmp;
; }
;
; Check if the expanded SAI are created
;
; CHECK: double MemRef_B_Stmt_for_body3_expanded[2000][3000]; // Element size 8
;
; Check if the memory accesses are modified
;
; CHECK: new: { Stmt_for_body3[i0, i1] -> MemRef_B_Stmt_for_body3_expanded[i0, i1] };
; CHECK: new: { Stmt_for_end[i0] -> MemRef_B_Stmt_for_body3_expanded[i0, i0] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define double @mse(double* %A, double* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.end
  %indvars.iv3 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next4, %for.end ]
  br label %for.body3

for.body3:                                        ; preds = %for.body, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx = getelementptr inbounds double, double* %B, i64 %indvars.iv
  store double %conv, double* %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 3000
  br i1 %exitcond, label %for.body3, label %for.end

for.end:                                          ; preds = %for.body3
  %arrayidx5 = getelementptr inbounds double, double* %B, i64 %indvars.iv3
  %1 = bitcast double* %arrayidx5 to i64*
  %2 = load i64, i64* %1, align 8
  %arrayidx7 = getelementptr inbounds double, double* %A, i64 %indvars.iv3
  %3 = bitcast double* %arrayidx7 to i64*
  store i64 %2, i64* %3, align 8
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  %exitcond5 = icmp ne i64 %indvars.iv.next4, 2000
  br i1 %exitcond5, label %for.body, label %for.end10

for.end10:                                        ; preds = %for.end
  ret double 6.000000e+00
}
