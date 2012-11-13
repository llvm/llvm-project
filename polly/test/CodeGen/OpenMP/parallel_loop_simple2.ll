; RUN: opt %loadPolly %defaultOpts -polly-cloog -polly-codegen -enable-polly-openmp -analyze  < %s | FileCheck %s

;#define N 1024
;
;float C[N], X[N];
;
;float parallel_loop_simple2() {
;  int j;
;
;  for (j = 0; j < N; j++)
;    C[j] = j;
;
;  for (j = 0; j < N; j++)
;    X[j] += X[j];
;
;  return C[42] + X[42];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@C = common global [1024 x float] zeroinitializer, align 16
@X = common global [1024 x float] zeroinitializer, align 16

define float @parallel_loop_simple2() nounwind {
bb:
  br label %bb5

bb5:                                              ; preds = %bb7, %bb
  %indvar1 = phi i64 [ %indvar.next2, %bb7 ], [ 0, %bb ]
  %scevgep4 = getelementptr [1024 x float]* @C, i64 0, i64 %indvar1
  %j.0 = trunc i64 %indvar1 to i32
  %exitcond3 = icmp ne i64 %indvar1, 1024
  br i1 %exitcond3, label %bb6, label %bb8

bb6:                                              ; preds = %bb5
  %tmp = sitofp i32 %j.0 to float
  store float %tmp, float* %scevgep4, align 4
  br label %bb7

bb7:                                              ; preds = %bb6
  %indvar.next2 = add i64 %indvar1, 1
  br label %bb5

bb8:                                              ; preds = %bb5
  br label %bb9

bb9:                                              ; preds = %bb14, %bb8
  %indvar = phi i64 [ %indvar.next, %bb14 ], [ 0, %bb8 ]
  %scevgep = getelementptr [1024 x float]* @X, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %bb10, label %bb15

bb10:                                             ; preds = %bb9
  %tmp11 = load float* %scevgep, align 4
  %tmp12 = load float* %scevgep, align 4
  %tmp13 = fadd float %tmp12, %tmp11
  store float %tmp13, float* %scevgep, align 4
  br label %bb14

bb14:                                             ; preds = %bb10
  %indvar.next = add i64 %indvar, 1
  br label %bb9

bb15:                                             ; preds = %bb9
  %tmp16 = load float* getelementptr inbounds ([1024 x float]* @C, i64 0, i64 42), align 8
  %tmp17 = load float* getelementptr inbounds ([1024 x float]* @X, i64 0, i64 42), align 8
  %tmp18 = fadd float %tmp16, %tmp17
  ret float %tmp18
}

; CHECK: for (c2=0;c2<=1023;c2++) {
; CHECK:   Stmt_bb6(c2);
; CHECK: }
; CHECK: for (c2=0;c2<=1023;c2++) {
; CHECK:   Stmt_bb10(c2);
; CHECK: }
; CHECK: Parallel loop with iterator 'c2' generated
; CHECK: Parallel loop with iterator 'c2' generated
; CHECK-NOT: Parallel loop
