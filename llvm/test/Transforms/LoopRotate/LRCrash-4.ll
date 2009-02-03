; RUN: llvm-as < %s | opt -loop-rotate -disable-output

define void @InterpretSEIMessage(i8* %msg) {
entry:
	br label %bb15
bb6:		; preds = %bb15
	%gep.upgrd.1 = zext i32 %offset.1 to i64		; <i64> [#uses=1]
	%tmp11 = getelementptr i8* %msg, i64 %gep.upgrd.1		; <i8*> [#uses=0]
	br label %bb15
bb15:		; preds = %bb6, %entry
	%offset.1 = add i32 0, 1		; <i32> [#uses=2]
	br i1 false, label %bb6, label %bb17
bb17:		; preds = %bb15
	%offset.1.lcssa = phi i32 [ %offset.1, %bb15 ]		; <i32> [#uses=0]
	%payload_type.1.lcssa = phi i32 [ 0, %bb15 ]		; <i32> [#uses=0]
	ret void
}

