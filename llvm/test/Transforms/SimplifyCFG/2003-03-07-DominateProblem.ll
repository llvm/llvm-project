; RUN: llvm-as < %s | opt -simplifycfg -disable-output

define void @test(i32* %ldo, i1 %c, i1 %d) {
bb9:
	br i1 %c, label %bb11, label %bb10
bb10:		; preds = %bb9
	br label %bb11
bb11:		; preds = %bb10, %bb9
	%reg330 = phi i32* [ null, %bb10 ], [ %ldo, %bb9 ]		; <i32*> [#uses=1]
	br label %bb20
bb20:		; preds = %bb20, %bb11
	store i32* %reg330, i32** null
	br i1 %d, label %bb20, label %done
done:		; preds = %bb20
	ret void
}

