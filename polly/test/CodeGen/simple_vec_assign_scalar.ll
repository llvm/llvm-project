; RUN: opt %loadPolly %defaultOpts -polly-codegen %vector-opt -dce -S < %s | FileCheck %s

;#define N 1024
;float A[N];
;float B[N];
;
;void simple_vec_const(void) {
;  int i;
;
;  for (i = 0; i < 4; i++)
;    B[i] = A[i] + 1;
;}
;int main()
;{
;  simple_vec_const();
;  return A[42];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [1024 x float] zeroinitializer, align 16
@B = common global [1024 x float] zeroinitializer, align 16

define void @simple_vec_const() nounwind {
bb:
  br label %bb2

bb2:                                              ; preds = %bb5, %bb
  %indvar = phi i64 [ %indvar.next, %bb5 ], [ 0, %bb ]
  %scevgep = getelementptr [1024 x float]* @B, i64 0, i64 %indvar
  %scevgep1 = getelementptr [1024 x float]* @A, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 4
  br i1 %exitcond, label %bb3, label %bb6

bb3:                                              ; preds = %bb2
  %tmp = load float* %scevgep1, align 4
  %tmp4 = fadd float %tmp, 1.000000e+00
  store float %tmp4, float* %scevgep, align 4
  br label %bb5

bb5:                                              ; preds = %bb3
  %indvar.next = add i64 %indvar, 1
  br label %bb2

bb6:                                              ; preds = %bb2
  ret void
}

define i32 @main() nounwind {
bb:
  call void @simple_vec_const()
  %tmp = load float* getelementptr inbounds ([1024 x float]* @A, i64 0, i64 42), align 8
  %tmp1 = fptosi float %tmp to i32
  ret i32 %tmp1
}

; CHECK: %tmp4p_vec = fadd <4 x float> %tmp_p_vec_full, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>

