; RUN: opt -polly-canonicalize %loadPolly -polly-mse -analyze < %s | FileCheck %s
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

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define double @mse(double* %A, double* %B) {
entry:
  %A.addr = alloca double*, align 8
  %B.addr = alloca double*, align 8
  %i = alloca i32, align 4
  %tmp = alloca double, align 8
  %j = alloca i32, align 4
  store double* %A, double** %A.addr, align 8
  store double* %B, double** %B.addr, align 8
  store double 6.000000e+00, double* %tmp, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc8, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 2000
  br i1 %cmp, label %for.body, label %for.end10

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 3000
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load i32, i32* %j, align 4
  %conv = sitofp i32 %2 to double
  %3 = load double*, double** %B.addr, align 8
  %4 = load i32, i32* %j, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds double, double* %3, i64 %idxprom
  store double %conv, double* %arrayidx, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %5 = load i32, i32* %j, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  %6 = load double*, double** %B.addr, align 8
  %7 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %arrayidx5 = getelementptr inbounds double, double* %6, i64 %idxprom4
  %8 = load double, double* %arrayidx5, align 8
  %9 = load double*, double** %A.addr, align 8
  %10 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %10 to i64
  %arrayidx7 = getelementptr inbounds double, double* %9, i64 %idxprom6
  store double %8, double* %arrayidx7, align 8
  br label %for.inc8

for.inc8:                                         ; preds = %for.end
  %11 = load i32, i32* %i, align 4
  %inc9 = add nsw i32 %11, 1
  store i32 %inc9, i32* %i, align 4
  br label %for.cond

for.end10:                                        ; preds = %for.cond
  %12 = load double, double* %tmp, align 8
  ret double %12
}

