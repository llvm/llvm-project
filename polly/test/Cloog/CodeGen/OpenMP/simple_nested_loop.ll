; RUN: opt %loadPolly %defaultOpts -polly-codegen -enable-polly-openmp -verify-dom-info -S < %s | FileCheck %s

;#include <string.h>
;#define N 10
;
;double A[N];
;double B[N];
;
;void loop_openmp() {
;  for (int i = 0; i < N; i++) {
;    for (int j = 0; j < N; j++) {
;      A[j] += j;
;    }
;  }
;}
;
;int main () {
;  memset(A, 0, sizeof(float) * N);
;
;  loop_openmp();
;
;  return 0;
;}
;

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

@A = common global [10 x double] zeroinitializer, align 4
@B = common global [10 x double] zeroinitializer, align 4

define void @loop_openmp() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc12, %for.inc10 ]
  %exitcond1 = icmp ne i32 %i.0, 10
  br i1 %exitcond1, label %for.body, label %for.end13

for.body:                                         ; preds = %for.cond
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc, %for.body
  %tmp = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %arrayidx = getelementptr [10 x double]* @A, i32 0, i32 %tmp
  %exitcond = icmp ne i32 %tmp, 10
  br i1 %exitcond, label %for.body5, label %for.end

for.body5:                                        ; preds = %for.cond2
  %conv = sitofp i32 %tmp to double
  %tmp8 = load double* %arrayidx, align 4
  %add = fadd double %tmp8, %conv
  store double %add, double* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body5
  %inc = add nsw i32 %tmp, 1
  br label %for.cond2

for.end:                                          ; preds = %for.cond2
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %inc12 = add nsw i32 %i.0, 1
  br label %for.cond

for.end13:                                        ; preds = %for.cond
  ret void
}

define i32 @main() nounwind {
entry:
  call void @llvm.memset.p0i8.i32(i8* bitcast ([10 x double]* @A to i8*), i8 0, i32 40, i32 4, i1 false)
  call void @loop_openmp()
  ret i32 0
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

; CHECK: %omp.userContext = alloca { i32 }
; CHECK: getelementptr inbounds { i32 }* %omp.userContext, i32 0, i32 0
; CHECK: store i32 %polly.loopiv, i32* %1
; CHECK: %omp_data = bitcast { i32 }* %omp.userContext to i8*
; CHECK: call void @GOMP_parallel_loop_runtime_start(void (i8*)* @loop_openmp.omp_subfn, i8* %omp_data, i32 0, i32 0, i32 10, i32 1)
; CHECK: call void @loop_openmp.omp_subfn(i8* %omp_data)
; CHECK: call void @GOMP_parallel_end()

