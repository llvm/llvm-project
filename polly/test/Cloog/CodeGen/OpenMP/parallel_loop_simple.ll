; RUN: opt %loadPolly %defaultOpts -polly-cloog -polly-codegen -enable-polly-openmp -analyze  < %s | FileCheck %s

;#define M 1024
;#define N 1024
;#define K 1024
;
;float X[K];
;
;float parallel_loop_simple() {
;  int i, k;
;
;  for (i = 0; i < M; i++)
;    for (k = 0; k < K; k++)
;      X[k] += X[k];
;
;  return X[42];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@X = common global [1024 x float] zeroinitializer, align 16

define float @parallel_loop_simple() nounwind {
bb:
  br label %bb2

bb2:                                              ; preds = %bb10, %bb
  %i.0 = phi i32 [ 0, %bb ], [ %tmp11, %bb10 ]
  %exitcond1 = icmp ne i32 %i.0, 1024
  br i1 %exitcond1, label %bb3, label %bb12

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb8, %bb3
  %indvar = phi i64 [ %indvar.next, %bb8 ], [ 0, %bb3 ]
  %scevgep = getelementptr [1024 x float]* @X, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %bb5, label %bb9

bb5:                                              ; preds = %bb4
  %tmp = load float* %scevgep, align 4
  %tmp6 = load float* %scevgep, align 4
  %tmp7 = fadd float %tmp6, %tmp
  store float %tmp7, float* %scevgep, align 4
  br label %bb8

bb8:                                              ; preds = %bb5
  %indvar.next = add i64 %indvar, 1
  br label %bb4

bb9:                                              ; preds = %bb4
  br label %bb10

bb10:                                             ; preds = %bb9
  %tmp11 = add nsw i32 %i.0, 1
  br label %bb2

bb12:                                             ; preds = %bb2
  %tmp13 = load float* getelementptr inbounds ([1024 x float]* @X, i64 0, i64 42), align 8
  ret float %tmp13
}

; CHECK: for (c2=0;c2<=1023;c2++) {
; CHECK:   for (c4=0;c4<=1023;c4++) {
; CHECK:     Stmt_bb5(c2,c4);
; CHECK:   }
; CHECK: }
; CHECK: Parallel loop with iterator 'c4' generated
; CHECK-NOT: Parallel loop

