; RUN: opt %loadPolly -polly-canonicalize -polly-mse -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-canonicalize -polly-mse -pass-remarks-analysis="polly-mse" -analyze < %s 2>&1 | FileCheck %s --check-prefix=MSE
;
; Verify that Polly detects problems and does not expand the array
;
; Original source code :
;
; #define Ni 2000
; #define Nj 2000
; 
; double mse(double A[Ni], double B[Nj]) {
;   int i;
;   double tmp = 6;
;   for (i = 0; i < Ni; i++) {
;     B[i] = 2; 
;     for (int j = 0; j<Nj; j++) {
;       B[j] = j;
;     }
;     A[i] = B[i]; 
;   }
;   return tmp;
; }
;
; Check that the pass detects that there are more than 1 write access per array.
;
; MSE: MemRef_B has more than 1 write access.
;
; Check that the SAI is not expanded
;
; CHECK-NOT: double MemRef_B2_expanded[2000][3000]; // Element size 8
;
; Check that the  memory accesses are not modified
;
; CHECK-NOT: new: { Stmt_for_body3[i0, i1] -> MemRef_B_Stmt_for_body3_expanded[i0, i1] };
; CHECK-NOT: new: { Stmt_for_end[i0] -> MemRef_B_Stmt_for_body3_expanded

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

for.cond:                                         ; preds = %for.inc10, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 2000
  br i1 %cmp, label %for.body, label %for.end12

for.body:                                         ; preds = %for.cond
  %1 = load double*, double** %B.addr, align 8
  %2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds double, double* %1, i64 %idxprom
  store double 2.000000e+00, double* %arrayidx, align 8
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %3 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %3, 2000
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %4 = load i32, i32* %j, align 4
  %conv = sitofp i32 %4 to double
  %5 = load double*, double** %B.addr, align 8
  %6 = load i32, i32* %j, align 4
  %idxprom4 = sext i32 %6 to i64
  %arrayidx5 = getelementptr inbounds double, double* %5, i64 %idxprom4
  store double %conv, double* %arrayidx5, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %7 = load i32, i32* %j, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  %8 = load double*, double** %B.addr, align 8
  %9 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %9 to i64
  %arrayidx7 = getelementptr inbounds double, double* %8, i64 %idxprom6
  %10 = load double, double* %arrayidx7, align 8
  %11 = load double*, double** %A.addr, align 8
  %12 = load i32, i32* %i, align 4
  %idxprom8 = sext i32 %12 to i64
  %arrayidx9 = getelementptr inbounds double, double* %11, i64 %idxprom8
  store double %10, double* %arrayidx9, align 8
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %13 = load i32, i32* %i, align 4
  %inc11 = add nsw i32 %13, 1
  store i32 %inc11, i32* %i, align 4
  br label %for.cond

for.end12:                                        ; preds = %for.cond
  %14 = load double, double* %tmp, align 8
  ret double %14
}
