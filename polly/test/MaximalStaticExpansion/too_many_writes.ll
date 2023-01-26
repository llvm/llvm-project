; RUN: opt %loadPolly -polly-mse -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly "-passes=scop(print<polly-mse>)" -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-mse -polly-print-scops -pass-remarks-analysis="polly-mse" -disable-output < %s 2>&1 | FileCheck %s --check-prefix=MSE
; RUN: opt %loadNPMPolly "-passes=scop(print<polly-mse>)" -pass-remarks-analysis="polly-mse" -disable-output < %s 2>&1 | FileCheck %s --check-prefix=MSE
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
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define double @mse(ptr %A, ptr %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.end
  %indvars.iv3 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next4, %for.end ]
  %arrayidx = getelementptr inbounds double, ptr %B, i64 %indvars.iv3
  store double 2.000000e+00, ptr %arrayidx, align 8
  br label %for.body3

for.body3:                                        ; preds = %for.body, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx5 = getelementptr inbounds double, ptr %B, i64 %indvars.iv
  store double %conv, ptr %arrayidx5, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 2000
  br i1 %exitcond, label %for.body3, label %for.end

for.end:                                          ; preds = %for.body3
  %arrayidx7 = getelementptr inbounds double, ptr %B, i64 %indvars.iv3
  %1 = load i64, ptr %arrayidx7, align 8
  %arrayidx9 = getelementptr inbounds double, ptr %A, i64 %indvars.iv3
  store i64 %1, ptr %arrayidx9, align 8
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  %exitcond5 = icmp ne i64 %indvars.iv.next4, 2000
  br i1 %exitcond5, label %for.body, label %for.end12

for.end12:                                        ; preds = %for.end
  ret double 6.000000e+00
}
