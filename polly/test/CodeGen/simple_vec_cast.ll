; RUN: opt %loadPolly -basicaa -polly-codegen %vector-opt -dce -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [1024 x float] zeroinitializer, align 16
@B = common global [1024 x double] zeroinitializer, align 16

define void @simple_vec_const() nounwind {
bb:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %indvar = phi i64 [ %indvar.next, %bb3 ], [ 0, %bb ]
  %scevgep = getelementptr [1024 x double]* @B, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 4
  br i1 %exitcond, label %bb2, label %bb4

bb2:                                              ; preds = %bb1
  %tmp = load float* getelementptr inbounds ([1024 x float]* @A, i64 0, i64 0), align 16
  %tmp2 = fpext float %tmp to double
  store double %tmp2, double* %scevgep, align 4
  br label %bb3

bb3:                                              ; preds = %bb2
  %indvar.next = add i64 %indvar, 1
  br label %bb1

bb4:                                              ; preds = %bb1
  ret void
}

; CHECK:  fpext <4 x float> %tmp_p_splat to <4 x double>

