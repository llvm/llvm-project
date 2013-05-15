; RUN: opt %loadPolly %defaultOpts -polly-codegen -enable-polly-openmp -S  < %s | FileCheck %s

;#include <string.h>
;#include <stdio.h>
;#define N 5
;
;float A[N];
;float B[N];
;
;void loop1_openmp() {
;    for (int i = 0; i <= N; i++)
;      A[i] = 0;
;
;    for (int j = 0; j <= N; j++)
;      for (int k = 0; k <= N; k++)
;        B[k] += j;
;}
;
;int main () {
;  int i;
;  memset(A, 0, sizeof(float) * N);
;  memset(B, 0, sizeof(float) * N);
;
;  loop1_openmp();
;
;  return 0;
;}
;

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

@A = common global [5 x float] zeroinitializer, align 4
@B = common global [5 x float] zeroinitializer, align 4

define void @loop1_openmp() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %arrayidx = getelementptr [5 x float]* @A, i32 0, i32 %i.0
  %exitcond2 = icmp ne i32 %i.0, 6
  br i1 %exitcond2, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store float 0.000000e+00, float* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc21, %for.end
  %tmp = phi i32 [ 0, %for.end ], [ %inc23, %for.inc21 ]
  %exitcond1 = icmp ne i32 %tmp, 6
  br i1 %exitcond1, label %for.body7, label %for.end24

for.body7:                                        ; preds = %for.cond4
  br label %for.cond9

for.cond9:                                        ; preds = %for.inc17, %for.body7
  %k.0 = phi i32 [ 0, %for.body7 ], [ %inc19, %for.inc17 ]
  %arrayidx15 = getelementptr [5 x float]* @B, i32 0, i32 %k.0
  %exitcond = icmp ne i32 %k.0, 6
  br i1 %exitcond, label %for.body12, label %for.end20

for.body12:                                       ; preds = %for.cond9
  %conv = sitofp i32 %tmp to float
  %tmp16 = load float* %arrayidx15, align 4
  %add = fadd float %tmp16, %conv
  store float %add, float* %arrayidx15, align 4
  br label %for.inc17

for.inc17:                                        ; preds = %for.body12
  %inc19 = add nsw i32 %k.0, 1
  br label %for.cond9

for.end20:                                        ; preds = %for.cond9
  br label %for.inc21

for.inc21:                                        ; preds = %for.end20
  %inc23 = add nsw i32 %tmp, 1
  br label %for.cond4

for.end24:                                        ; preds = %for.cond4
  ret void
}

define i32 @main() nounwind {
entry:
  call void @llvm.memset.p0i8.i32(i8* bitcast ([5 x float]* @A to i8*), i8 0, i32 20, i32 4, i1 false)
  call void @llvm.memset.p0i8.i32(i8* bitcast ([5 x float]* @B to i8*), i8 0, i32 20, i32 4, i1 false)
  call void @loop1_openmp()
  ret i32 0
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

; CHECK: %omp.userContext = alloca {}
; CHECK: %omp.userContext1 = alloca { i32 }

