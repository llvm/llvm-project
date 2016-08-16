; RUN: opt %loadPolly -basicaa -polly-codegen -polly-vectorizer=polly -S \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [1024 x i32] zeroinitializer, align 16
@B = common global [1024 x i32] zeroinitializer, align 16

declare i32 @foo(i32) readnone

define void @force_alignment() nounwind {
;CHECK: @force_alignment
entry:
  br label %body

body:
  %indvar = phi i64 [ 0, %entry ], [ %indvar_next, %body ]
  %scevgep = getelementptr [1024 x i32], [1024 x i32]* @B, i64 0, i64 %indvar
; CHECK: [[T2:%.load]] = load i32, i32* getelementptr inbounds ([1024 x i32], [1024 x i32]* @A, i32 0, i32 0), align 4
; CHECK: %value_p.splatinsert = insertelement <4 x i32> undef, i32 [[T2]], i32 0
  %value = load i32, i32* getelementptr inbounds ([1024 x i32], [1024 x i32]* @A, i64 0, i64 0), align 4
  %result = tail call i32 @foo(i32 %value) nounwind
  store i32 %result, i32* %scevgep, align 4
  %indvar_next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar_next, 4
  br i1 %exitcond, label %return, label %body

return:
  ret void
}

