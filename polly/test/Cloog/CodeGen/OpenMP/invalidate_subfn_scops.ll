; RUN: opt %loadPolly %defaultOpts -polly-codegen -enable-polly-openmp -analyze < %s -debug-only=polly-detect 2>&1 | not FileCheck %s

;#define N 500000
;float A[N];
;int main() {
;  int j, k;
;
;  for(k = 0; k < N; k++)
;    for (j = 0; j <= N; j++)
;      A[j] = k;
;
;  return 0;
;}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

@A = common global [500000 x float] zeroinitializer, align 4

define i32 @main() nounwind {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc8, %entry.split
  %0 = phi i32 [ 0, %entry.split ], [ %inc10, %for.inc8 ]
  br label %for.body4

for.body4:                                        ; preds = %for.body4, %for.cond1.preheader
  %j.01 = phi i32 [ 0, %for.cond1.preheader ], [ %inc, %for.body4 ]
  %arrayidx = getelementptr [500000 x float]* @A, i32 0, i32 %j.01
  %conv = sitofp i32 %0 to float
  store float %conv, float* %arrayidx, align 4
  %inc = add nsw i32 %j.01, 1
  %exitcond = icmp eq i32 %inc, 500001
  br i1 %exitcond, label %for.inc8, label %for.body4

for.inc8:                                         ; preds = %for.body4
  %inc10 = add nsw i32 %0, 1
  %exitcond3 = icmp eq i32 %inc10, 500000
  br i1 %exitcond3, label %for.end11, label %for.cond1.preheader

for.end11:                                        ; preds = %for.inc8
  ret i32 0
}


; CHECK: Checking region: omp.setup
