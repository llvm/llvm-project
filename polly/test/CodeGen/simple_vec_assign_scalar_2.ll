; RUN: opt %loadPolly -basic-aa -polly-codegen -polly-vectorizer=polly -dce -S < %s | FileCheck %s

;#define N 1024
;float A[N];
;float B[N];
;
;void simple_vec_const(void) {
;  int i;
;
;  for (i = 0; i < 4; i++)
;    B[i] = A[i] + i;
;}
;int main()
;{
;  simple_vec_const();
;  return A[42];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x float] zeroinitializer, align 16
@B = common global [1024 x float] zeroinitializer, align 16

define void @simple_vec_const() nounwind {
bb:
  br label %bb2

bb2:                                              ; preds = %bb6, %bb
  %indvar = phi i64 [ %indvar.next, %bb6 ], [ 0, %bb ]
  %scevgep = getelementptr [1024 x float], [1024 x float]* @B, i64 0, i64 %indvar
  %i.0 = trunc i64 %indvar to i32
  %scevgep1 = getelementptr [1024 x float], [1024 x float]* @A, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 4
  br i1 %exitcond, label %bb3, label %bb7

bb3:                                              ; preds = %bb2
  %tmp = load float, float* %scevgep1, align 4
  %tmp4 = sitofp i32 %i.0 to float
  %tmp5 = fadd float %tmp, %tmp4
  store float %tmp5, float* %scevgep, align 4
  br label %bb6

bb6:                                              ; preds = %bb3
  %indvar.next = add i64 %indvar, 1
  br label %bb2

bb7:                                              ; preds = %bb2
  ret void
}

define i32 @main() nounwind {
bb:
  call void @simple_vec_const()
  %tmp = load float, float* getelementptr inbounds ([1024 x float], [1024 x float]* @A, i64 0, i64 42), align 8
  %tmp1 = fptosi float %tmp to i32
  ret i32 %tmp1
}


; CHECK: insertelement <4 x float> undef, float %{{[^,]+}}, i32 0
; CHECK: insertelement <4 x float> %0, float %{{[^,]+}}, i32 1
; CHECK: insertelement <4 x float> %1, float %{{[^,]+}}, i32 2
; CHECK: insertelement <4 x float> %2, float %{{[^,]+}}, i32 3
; CHECK: fadd <4 x float> %tmp_p_vec_full, %3

