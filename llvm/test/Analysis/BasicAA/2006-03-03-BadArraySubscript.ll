; RUN: llvm-as < %s | opt -aa-eval -disable-output |& grep {2 no alias respon}
; TEST that A[1][0] may alias A[0][i].

define void @test(i32 %N) {
entry:
	%X = alloca [3 x [3 x i32]]		; <[3 x [3 x i32]]*> [#uses=4]
	%tmp.24 = icmp sgt i32 %N, 0		; <i1> [#uses=1]
	br i1 %tmp.24, label %no_exit, label %loopexit

no_exit:		; preds = %no_exit, %entry
	%i.0.0 = phi i32 [ 0, %entry ], [ %inc, %no_exit ]		; <i32> [#uses=2]
	%tmp.6 = getelementptr [3 x [3 x i32]]* %X, i32 0, i32 0, i32 %i.0.0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp.6
	%tmp.8 = getelementptr [3 x [3 x i32]]* %X, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp.9 = load i32* %tmp.8		; <i32> [#uses=1]
	%tmp.11 = getelementptr [3 x [3 x i32]]* %X, i32 0, i32 1, i32 0		; <i32*> [#uses=1]
	%tmp.12 = load i32* %tmp.11		; <i32> [#uses=1]
	%tmp.13 = add i32 %tmp.12, %tmp.9		; <i32> [#uses=1]
	%inc = add i32 %i.0.0, 1		; <i32> [#uses=2]
	%tmp.2 = icmp slt i32 %inc, %N		; <i1> [#uses=1]
	br i1 %tmp.2, label %no_exit, label %loopexit

loopexit:		; preds = %no_exit, %entry
	%Y.0.1 = phi i32 [ 0, %entry ], [ %tmp.13, %no_exit ]		; <i32> [#uses=1]
	%tmp.4 = getelementptr [3 x [3 x i32]]* %X, i32 0, i32 0		; <[3 x i32]*> [#uses=1]
	%tmp.15 = call i32 (...)* @foo( [3 x i32]* %tmp.4, i32 %Y.0.1 )		; <i32> [#uses=0]
	ret void
}

declare i32 @foo(...)
