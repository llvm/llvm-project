; RUN: opt -opaque-pointers=0 %loadPolly -basic-aa -polly-codegen -polly-vectorizer=polly -S \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s

;#define N 1024
;float A[N];
;float B[N];
;
;void simple_vec_const(void) {
;  int i;
;
;  for (i = 0; i < 4; i++)
;    B[i] = A[0];
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
; <label>:0
  br label %1

; <label>:1                                       ; preds = %4, %0
  %indvar = phi i64 [ %indvar.next, %4 ], [ 0, %0 ]
  %scevgep = getelementptr [1024 x float], [1024 x float]* @B, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 4
  br i1 %exitcond, label %2, label %5

; <label>:2                                       ; preds = %1
  %3 = load float, float* getelementptr inbounds ([1024 x float], [1024 x float]* @A, i64 0, i64 0), align 16
  store float %3, float* %scevgep, align 4
  br label %4

; <label>:4                                       ; preds = %2
  %indvar.next = add i64 %indvar, 1
  br label %1

; <label>:5                                       ; preds = %1
  ret void
}

define i32 @main() nounwind {
  call void @simple_vec_const()
  %1 = load float, float* getelementptr inbounds ([1024 x float], [1024 x float]* @A, i64 0, i64 42), align 8
  %2 = fptosi float %1 to i32
  ret i32 %2
}


; CHECK:   %.load = load float, float* getelementptr inbounds ([1024 x float], [1024 x float]* @A, i32 0, i32 0)

; CHECK: polly.stmt.:                                      ; preds = %polly.start
; CHECK:   %_p.splatinsert = insertelement <4 x float> poison, float %.load, i64 0
; CHECK:   %_p.splat = shufflevector <4 x float> %_p.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
