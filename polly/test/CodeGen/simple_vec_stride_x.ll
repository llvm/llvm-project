; RUN: opt %loadPolly -basic-aa -polly-codegen -polly-vectorizer=polly  -dce -S < %s | FileCheck %s

;#define N 1024
;float A[N];
;float B[N];
;
;void simple_vec_stride_x(void) {
;  int i;
;
;  for (i = 0; i < 4; i++)
;    B[2 * i] = A[2 * i];
;}
;int main()
;{
;  simple_vec_stride_x();
;  return A[42];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x float] zeroinitializer, align 16
@B = common global [1024 x float] zeroinitializer, align 16

define void @simple_vec_stride_x() nounwind {
bb:
  br label %bb2

bb2:                                              ; preds = %bb5, %bb
  %indvar = phi i64 [ %indvar.next, %bb5 ], [ 0, %bb ]
  %tmp = mul i64 %indvar, 2
  %scevgep = getelementptr [1024 x float], ptr @B, i64 0, i64 %tmp
  %scevgep1 = getelementptr [1024 x float], ptr @A, i64 0, i64 %tmp
  %exitcond = icmp ne i64 %indvar, 4
  br i1 %exitcond, label %bb3, label %bb6

bb3:                                              ; preds = %bb2
  %tmp4 = load float, ptr %scevgep1, align 8
  store float %tmp4, ptr %scevgep, align 8
  br label %bb5

bb5:                                              ; preds = %bb3
  %indvar.next = add i64 %indvar, 1
  br label %bb2

bb6:                                              ; preds = %bb2
  ret void
}

define i32 @main() nounwind {
bb:
  call void @simple_vec_stride_x()
  %tmp = load float, ptr getelementptr inbounds ([1024 x float], ptr @A, i64 0, i64 42), align 8
  %tmp1 = fptosi float %tmp to i32
  ret i32 %tmp1
}

; CHECK: [[LOAD1:%[a-zA-Z0-9_]+_scalar_]] = load float, ptr
; CHECK: [[VEC1:%[a-zA-Z0-9_]+]] = insertelement <4 x float> undef, float [[LOAD1]], i32 0
; CHECK: [[LOAD2:%[a-zA-Z0-9_]+]] = load float, ptr
; CHECK: [[VEC2:%[a-zA-Z0-9_]+]] = insertelement <4 x float> [[VEC1]], float [[LOAD2]], i32 1
; CHECK: [[LOAD3:%[a-zA-Z0-9_]+]] = load float, ptr
; CHECK: [[VEC3:%[a-zA-Z0-9_]+]] = insertelement <4 x float> [[VEC2]], float [[LOAD3]], i32 2
; CHECK: [[LOAD4:%[a-zA-Z0-9_]+]] = load float, ptr
; CHECK: [[VEC4:%[a-zA-Z0-9_]+]] = insertelement <4 x float> [[VEC3]], float [[LOAD4]], i32 3
; CHECK: [[EL1:%[a-zA-Z0-9_]+]] = extractelement <4 x float> [[VEC4]], i32 0
; CHECK: store float [[EL1]]
; CHECK: [[EL2:%[a-zA-Z0-9_]+]] = extractelement <4 x float> [[VEC4]], i32 1
; CHECK: store float [[EL2]]
; CHECK: [[EL3:%[a-zA-Z0-9_]+]] = extractelement <4 x float> [[VEC4]], i32 2
; CHECK: store float [[EL3]]
; CHECK: [[EL4:%[a-zA-Z0-9_]+]] = extractelement <4 x float> [[VEC4]], i32 3
; CHECK: store float [[EL4]]
