; RUN: opt %loadPolly %defaultOpts -mem2reg -polly-codegen -enable-polly-openmp -S %s

;#include <string.h>
;#define N 10240000
;
;float A[N];
;float B[N];
;
;void loop1_openmp() {
;	for (int i = 0; i <= N; i++)
;		A[i] = 0;
;	for (int j = 0; j <= N; j++)
;		B[j] = 0;
;}
;
;
;int main () {
;  int i;
;  memset(A, 0, sizeof(float) * N);
;  memset(B, 1, sizeof(float) * N);
;
;  loop1_openmp();
;
;  return 0;
;}
;

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

@A = common global [10240000 x float] zeroinitializer, align 4
@B = common global [10240000 x float] zeroinitializer, align 4

define void @loop1_openmp() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %arrayidx = getelementptr [10240000 x float]* @A, i32 0, i32 %i.0
  %exitcond1 = icmp ne i32 %i.0, 10240001
  br i1 %exitcond1, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store float 0.000000e+00, float* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc10, %for.end
  %j.0 = phi i32 [ 0, %for.end ], [ %inc12, %for.inc10 ]
  %arrayidx9 = getelementptr [10240000 x float]* @B, i32 0, i32 %j.0
  %exitcond = icmp ne i32 %j.0, 10240001
  br i1 %exitcond, label %for.body7, label %for.end13

for.body7:                                        ; preds = %for.cond4
  store float 0.000000e+00, float* %arrayidx9, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %for.body7
  %inc12 = add nsw i32 %j.0, 1
  br label %for.cond4

for.end13:                                        ; preds = %for.cond4
  ret void
}

define i32 @main() nounwind {
entry:
  call void @llvm.memset.p0i8.i32(i8* bitcast ([10240000 x float]* @A to i8*), i8 0, i32 40960000, i32 4, i1 false)
  call void @llvm.memset.p0i8.i32(i8* bitcast ([10240000 x float]* @B to i8*), i8 1, i32 40960000, i32 4, i1 false)
  call void @loop1_openmp()
  ret i32 0
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind
