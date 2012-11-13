; RUN: opt %loadPolly -basicaa -polly-codegen %vector-opt -S -polly-codegen-scev=false < %s | FileCheck %s
; RUN: opt %loadPolly -basicaa -polly-codegen %vector-opt -S -polly-codegen-scev=true < %s | FileCheck %s -check-prefix=CHECK-SCEV
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [1024 x float**] zeroinitializer, align 16
@B = common global [1024 x float**] zeroinitializer, align 16

declare float @foo(float) readnone

define void @simple_vec_call() nounwind {
entry:
  br label %body

body:
  %indvar = phi i64 [ 0, %entry ], [ %indvar_next, %body ]
  %scevgep = getelementptr [1024 x float**]* @B, i64 0, i64 %indvar
  %value = load float*** getelementptr inbounds ([1024 x float**]* @A, i64 0, i64 0), align 16
  store float** %value, float*** %scevgep, align 4
  %indvar_next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar_next, 4
  br i1 %exitcond, label %return, label %body

return:
  ret void
}

; CHECK: %p_scevgep = getelementptr [1024 x float**]* @B, i64 0, i64 0
; CHECK: %p_scevgep1 = getelementptr [1024 x float**]* @B, i64 0, i64 1
; CHECK: %p_scevgep2 = getelementptr [1024 x float**]* @B, i64 0, i64 2
; CHECK: %p_scevgep3 = getelementptr [1024 x float**]* @B, i64 0, i64 3
; CHECK: %value_p_scalar_ = load float*** getelementptr inbounds ([1024 x float**]* @A, i64 0, i64 0)
; CHECK: %value_p_scalar_4 = load float*** getelementptr inbounds ([1024 x float**]* @A, i64 0, i64 0)
; CHECK: %value_p_scalar_5 = load float*** getelementptr inbounds ([1024 x float**]* @A, i64 0, i64 0)
; CHECK: %value_p_scalar_6 = load float*** getelementptr inbounds ([1024 x float**]* @A, i64 0, i64 0)
; CHECK: store float** %value_p_scalar_, float*** %p_scevgep, align 4
; CHECK: store float** %value_p_scalar_4, float*** %p_scevgep1, align 4
; CHECK: store float** %value_p_scalar_5, float*** %p_scevgep2, align 4
; CHECK: store float** %value_p_scalar_6, float*** %p_scevgep3, align 4

; CHECK-SCEV: %value_p_scalar_ = load float*** getelementptr inbounds ([1024 x float**]* @A, i64 0, i64 0)
; CHECK-SCEV: %value_p_scalar_1 = load float*** getelementptr inbounds ([1024 x float**]* @A, i64 0, i64 0)
; CHECK-SCEV: %value_p_scalar_2 = load float*** getelementptr inbounds ([1024 x float**]* @A, i64 0, i64 0)
; CHECK-SCEV: %value_p_scalar_3 = load float*** getelementptr inbounds ([1024 x float**]* @A, i64 0, i64 0)
; CHECK-SCEV: store float** %value_p_scalar_, float*** getelementptr inbounds ([1024 x float**]* @B, i64 0, i64 0), align 4
; CHECK-SCEV: store float** %value_p_scalar_1, float*** getelementptr inbounds ([1024 x float**]* @B, i64 0, i64 1), align 4
; CHECK-SCEV: store float** %value_p_scalar_2, float*** getelementptr inbounds ([1024 x float**]* @B, i64 0, i64 2), align 4
; CHECK-SCEV: store float** %value_p_scalar_3, float*** getelementptr inbounds ([1024 x float**]* @B, i64 0, i64 3), align 4
