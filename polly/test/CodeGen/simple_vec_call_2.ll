; RUN: opt -opaque-pointers=0 %loadPolly -basic-aa -polly-codegen -polly-vectorizer=polly -dce \
; RUN: -polly-invariant-load-hoisting=true -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x float] zeroinitializer, align 16
@B = common global [1024 x float**] zeroinitializer, align 16

declare float** @foo(float) readnone

define void @simple_vec_call() nounwind {
entry:
  br label %body

body:
  %indvar = phi i64 [ 0, %entry ], [ %indvar_next, %body ]
  %scevgep = getelementptr [1024 x float**], [1024 x float**]* @B, i64 0, i64 %indvar
  %value = load float, float* getelementptr inbounds ([1024 x float], [1024 x float]* @A, i64 0, i64 0), align 16
  %result = tail call float** @foo(float %value) nounwind
  store float** %result, float*** %scevgep, align 4
  %indvar_next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar_next, 4
  br i1 %exitcond, label %return, label %body

return:
  ret void
}

; CHECK: [[RES1:%[a-zA-Z0-9_]+]] = tail call float** @foo(float %.load) [[NUW:#[0-9]+]]
; CHECK: [[RES2:%[a-zA-Z0-9_]+]] = tail call float** @foo(float %.load) [[NUW]]
; CHECK: [[RES3:%[a-zA-Z0-9_]+]] = tail call float** @foo(float %.load) [[NUW]]
; CHECK: [[RES4:%[a-zA-Z0-9_]+]] = tail call float** @foo(float %.load) [[NUW]]
; CHECK: %0 = insertelement <4 x float**> undef, float** %p_result, i32 0
; CHECK: %1 = insertelement <4 x float**> %0, float** %p_result1, i32 1
; CHECK: %2 = insertelement <4 x float**> %1, float** %p_result2, i32 2
; CHECK: %3 = insertelement <4 x float**> %2, float** %p_result3, i32 3
; CHECK: store <4 x float**> %3, <4 x float**>* bitcast ([1024 x float**]* @B to <4 x float**>*), align
; CHECK: attributes [[NUW]] = { nounwind }
