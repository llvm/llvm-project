; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -disable-output 2>&1 | FileCheck %s
; TEST that A[1][0] may alias A[0][i].
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; CHECK: 2 no alias responses

define void @test(i32 %N) {
entry:
	%X = alloca [3 x [3 x i32]]		; <ptr> [#uses=4]
	%tmp.24 = icmp sgt i32 %N, 0		; <i1> [#uses=1]
	br i1 %tmp.24, label %no_exit, label %loopexit

no_exit:		; preds = %no_exit, %entry
	%i.0.0 = phi i32 [ 0, %entry ], [ %inc, %no_exit ]		; <i32> [#uses=2]
	%tmp.6 = getelementptr [3 x [3 x i32]], ptr %X, i32 0, i32 0, i32 %i.0.0		; <ptr> [#uses=1]
	store i32 1, ptr %tmp.6
	%tmp.8 = getelementptr [3 x [3 x i32]], ptr %X, i32 0, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp.9 = load i32, ptr %tmp.8		; <i32> [#uses=1]
	%tmp.11 = getelementptr [3 x [3 x i32]], ptr %X, i32 0, i32 1, i32 0		; <ptr> [#uses=1]
	%tmp.12 = load i32, ptr %tmp.11		; <i32> [#uses=1]
	%tmp.13 = add i32 %tmp.12, %tmp.9		; <i32> [#uses=1]
	%inc = add i32 %i.0.0, 1		; <i32> [#uses=2]
	%tmp.2 = icmp slt i32 %inc, %N		; <i1> [#uses=1]
	br i1 %tmp.2, label %no_exit, label %loopexit

loopexit:		; preds = %no_exit, %entry
	%Y.0.1 = phi i32 [ 0, %entry ], [ %tmp.13, %no_exit ]		; <i32> [#uses=1]
	%tmp.4 = getelementptr [3 x [3 x i32]], ptr %X, i32 0, i32 0		; <ptr> [#uses=1]
  load [3 x i32], ptr %tmp.4
	%tmp.15 = call i32 (...) @foo( ptr %tmp.4, i32 %Y.0.1 )		; <i32> [#uses=0]
	ret void
}

declare i32 @foo(...)
