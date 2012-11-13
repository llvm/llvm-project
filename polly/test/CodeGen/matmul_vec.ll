; RUN: opt %loadPolly %defaultOpts -polly-import-jscop -polly-import-jscop-dir=%S -polly-codegen %vector-opt -S -dce < %s | FileCheck %s

;#define M 1024
;#define N 1024
;#define K 1024
;float A[K][M];
;float B[N][K];
;float C[M][N];
;/*
;void matmul_vec(void) {
;  int i, j, k;
;
;
;  /* With much unrolling
;  for (i=0;i<=M;i++)
;    for (j=0;j<=N;j+=4)
;      for (k=0;k<=K;k+=8)
;        for (kk=k;kk<=k+7;kk++)
;          for (jj=j;jj<=j+3;jj++)
;            C[i][jj] += A[kk][i] * B[jj][kk];
;            vec_load    splat      scalar_load
;   */
;  /* Without unrolling
;  for (i=0;i<=M;i++)
;    for (j=0;j<=N;j+=4)
;      for (k=0;k<=K;k++)
;          for (jj=j;jj<=j+3;jj++)
;            C[i][jj] += A[k][i] * B[jj][kk];
;            vec_load    splat      scalar_load
;   /
;
;}
;i*/
;int main()
;{
;  int i, j, k;
;  //matmul_vec();
;  for(i=0; i<M/4; i++)
;    for(k=0; k<K; k++) {
;      for(j=0; j<N; j++)
;        C[i+0][j] += A[k][i+0] * B[j][k];
;        C[i+1][j] += A[k][i+1] * B[j][k];
;        C[i+2][j] += A[k][i+2] * B[j][k];
;        C[i+3][j] += A[k][i+3] * B[j][k];
;      }
;
;  return A[42][42];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [1024 x [1024 x float]] zeroinitializer, align 16
@B = common global [1024 x [1024 x float]] zeroinitializer, align 16
@C = common global [1024 x [1024 x float]] zeroinitializer, align 16

define void @matmul_vec() nounwind {
; <label>:0
  br label %1

; <label>:1                                       ; preds = %16, %0
  %indvar3 = phi i64 [ %indvar.next4, %16 ], [ 0, %0 ]
  %exitcond9 = icmp ne i64 %indvar3, 1024
  br i1 %exitcond9, label %2, label %17

; <label>:2                                       ; preds = %1
  br label %3

; <label>:3                                       ; preds = %14, %2
  %indvar1 = phi i64 [ %indvar.next2, %14 ], [ 0, %2 ]
  %scevgep8 = getelementptr [1024 x [1024 x float]]* @C, i64 0, i64 %indvar3, i64 %indvar1
  %exitcond6 = icmp ne i64 %indvar1, 1024
  br i1 %exitcond6, label %4, label %15

; <label>:4                                       ; preds = %3
  br label %5

; <label>:5                                       ; preds = %12, %4
  %indvar = phi i64 [ %indvar.next, %12 ], [ 0, %4 ]
  %scevgep5 = getelementptr [1024 x [1024 x float]]* @A, i64 0, i64 %indvar, i64 %indvar3
  %scevgep = getelementptr [1024 x [1024 x float]]* @B, i64 0, i64 %indvar1, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %6, label %13

; <label>:6                                       ; preds = %5
  %7 = load float* %scevgep5, align 4
  %8 = load float* %scevgep, align 4
  %9 = fmul float %7, %8
  %10 = load float* %scevgep8, align 4
  %11 = fadd float %10, %9
  store float %11, float* %scevgep8, align 4
  br label %12

; <label>:12                                      ; preds = %6
  %indvar.next = add i64 %indvar, 1
  br label %5

; <label>:13                                      ; preds = %5
  br label %14

; <label>:14                                      ; preds = %13
  %indvar.next2 = add i64 %indvar1, 1
  br label %3

; <label>:15                                      ; preds = %3
  br label %16

; <label>:16                                      ; preds = %15
  %indvar.next4 = add i64 %indvar3, 1
  br label %1

; <label>:17                                      ; preds = %1
  ret void
}

define i32 @main() nounwind {
  call void @matmul_vec()
  %1 = load float* getelementptr inbounds ([1024 x [1024 x float]]* @A, i64 0, i64 42, i64 42), align 8
  %2 = fptosi float %1 to i32
  ret i32 %2
}

; CHECK: load <1 x float>*
; CHECK: shufflevector <1 x float>
; CHECK: load float*
; CHECK: insertelement <4 x float>
; CHECK: load float*
; CHECK: insertelement <4 x float>
; CHECK: load float*
; CHECK: insertelement <4 x float>
; CHECK: load float*
; CHECK: insertelement <4 x float>
; CHECK: fmul <4 x float>
; CHECK: bitcast float*
; CHECK: load <4 x float>*
; CHECK: fadd <4 x float>
; CHECK: bitcast float*
; CHECK: store <4 x float>
