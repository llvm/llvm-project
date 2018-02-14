; RUN: opt %loadPolly -polly-mse -analyze < %s | FileCheck %s
;
; Verify that the accesses are correctly expanded for MemoryKind::Value
;
; Original source code :
;
; #define Ni 10000
; #define Nj 10000
;
; void mse(double A[Ni], double B[Nj]) {
;   int i,j;
;   double tmp = 6;
;   for (i = 0; i < Ni; i++) {
;     tmp = i;
;     for (int j = 0; j<Nj; j++) {
;       A[j] = tmp+3;
;     }
;     B[i] = tmp;
;   }
; }
;
; Check if the expanded SAI are created
;
; CHECK: double MemRef_conv_Stmt_for_body_expanded[10000]; // Element size 8
;
; Check if the memory accesses are modified
;
; CHECK: new: { Stmt_for_body[i0] -> MemRef_conv_Stmt_for_body_expanded[i0] };
; CHECK: new: { Stmt_for_body5[i0, i1] -> MemRef_conv_Stmt_for_body_expanded[i0] };
; CHECK: new: { Stmt_for_end[i0] -> MemRef_conv_Stmt_for_body_expanded[i0] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @mse(double* %A, double* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.end
  %indvars.iv3 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next4, %for.end ]
  %0 = trunc i64 %indvars.iv3 to i32
  %conv = sitofp i32 %0 to double
  br label %for.body5

for.body5:                                        ; preds = %for.body, %for.body5
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body5 ]
  %add = fadd double %conv, 3.000000e+00
  %arrayidx = getelementptr inbounds double, double* %A, i64 %indvars.iv
  store double %add, double* %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 10000
  br i1 %exitcond, label %for.body5, label %for.end

for.end:                                          ; preds = %for.body5
  %arrayidx7 = getelementptr inbounds double, double* %B, i64 %indvars.iv3
  store double %conv, double* %arrayidx7, align 8
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  %exitcond5 = icmp ne i64 %indvars.iv.next4, 10000
  br i1 %exitcond5, label %for.body, label %for.end10

for.end10:                                        ; preds = %for.end
  ret void
}
