; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin | grep align | count 1

@A = common global [100 x i32] zeroinitializer, align 32		; <[100 x i32]*> [#uses=1]

define i8* @test(i8* %Q, i32* %L) nounwind {
entry:
	%tmp = tail call i32 (...)* @foo() nounwind		; <i32> [#uses=2]
	%tmp1 = inttoptr i32 %tmp to i8*		; <i8*> [#uses=1]
	br label %bb1

bb:		; preds = %bb1, %bb1
	%indvar.next = add i32 %P.0.rec, 1		; <i32> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%P.0.rec = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%P.0 = getelementptr i8* %tmp1, i32 %P.0.rec		; <i8*> [#uses=3]
	%tmp2 = load i8* %P.0, align 1		; <i8> [#uses=1]
	switch i8 %tmp2, label %bb4 [
		i8 12, label %bb
		i8 42, label %bb
	]

bb4:		; preds = %bb1
	%tmp3 = ptrtoint i8* %P.0 to i32		; <i32> [#uses=1]
	%tmp4 = sub i32 %tmp3, %tmp		; <i32> [#uses=1]
	%tmp5 = getelementptr [100 x i32]* @A, i32 0, i32 %tmp4		; <i32*> [#uses=1]
	store i32 4, i32* %tmp5, align 4
	ret i8* %P.0
}

declare i32 @foo(...)
