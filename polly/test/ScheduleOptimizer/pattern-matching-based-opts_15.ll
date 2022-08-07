; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -debug-only=polly-opt-isl -disable-output \
; RUN: -polly-tc-opt=true < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
;    for (i = 0; i < _PB_NI; i++)
;      for (j = 0; j < _PB_NJ; j++)
;      {
;        for (k = 0; k < _PB_NK; k++)
;        {
;          double Mul = A[i][k] * B[k][j];
;          D[i][j][k] += Mul;
;          C[i][j] += Mul;
;        }
;      }
;
; CHECK-NOT: The matrix multiplication pattern was detected
; CHECK-NOT: The tensor contraction pattern was detected

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @kernel_gemm([1024 x double]* %C, [1024 x double]* %A, [1024 x double]* %B, [1024 x [1024 x double]]* %D) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc27, %entry
  %indvars.iv7 = phi i64 [ 0, %entry ], [ %indvars.iv.next8, %for.inc27 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc24, %for.cond1.preheader
  %indvars.iv4 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next5, %for.inc24 ]
  %arrayidx22 = getelementptr inbounds [1024 x double], [1024 x double]* %C, i64 %indvars.iv7, i64 %indvars.iv4
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.cond4.preheader
  %indvars.iv = phi i64 [ 0, %for.cond4.preheader ], [ %indvars.iv.next, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [1024 x double], [1024 x double]* %A, i64 %indvars.iv7, i64 %indvars.iv
  %i = load double, double* %arrayidx8, align 8
  %arrayidx12 = getelementptr inbounds [1024 x double], [1024 x double]* %B, i64 %indvars.iv, i64 %indvars.iv4
  %i1 = load double, double* %arrayidx12, align 8
  %mul = fmul double %i1, %i
  %arrayidx18 = getelementptr inbounds [1024 x [1024 x double]], [1024 x [1024 x double]]* %D, i64 %indvars.iv7, i64 %indvars.iv4, i64 %indvars.iv
  %i2 = load double, double* %arrayidx18, align 8
  %add = fadd double %i2, %mul
  store double %add, double* %arrayidx18, align 8
  %i3 = load double, double* %arrayidx22, align 8
  %add23 = fadd double %i3, %mul
  store double %add23, double* %arrayidx22, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.inc24, label %for.body6

for.inc24:                                        ; preds = %for.body6
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1
  %exitcond6.not = icmp eq i64 %indvars.iv.next5, 1024
  br i1 %exitcond6.not, label %for.inc27, label %for.cond4.preheader

for.inc27:                                        ; preds = %for.inc24
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1
  %exitcond9.not = icmp eq i64 %indvars.iv.next8, 1024
  br i1 %exitcond9.not, label %for.end29, label %for.cond1.preheader

for.end29:                                        ; preds = %for.inc27
  ret void
}
