; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output

define void @test(ptr %ldo, i1 %c, i1 %d) {
bb9:
	br i1 %c, label %bb11, label %bb10
bb10:		; preds = %bb9
	br label %bb11
bb11:		; preds = %bb10, %bb9
	%reg330 = phi ptr [ null, %bb10 ], [ %ldo, %bb9 ]		; <ptr> [#uses=1]
	br label %bb20
bb20:		; preds = %bb20, %bb11
	store ptr %reg330, ptr null
	br i1 %d, label %bb20, label %done
done:		; preds = %bb20
	ret void
}

