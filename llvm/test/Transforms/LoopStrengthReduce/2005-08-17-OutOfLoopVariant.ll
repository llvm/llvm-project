; RUN: opt < %s -loop-reduce -disable-output

define i32 @image_to_texture(i32 %indvar454) {
loopentry.1.outer:
	%j.2.1.ph = bitcast i32 %indvar454 to i32		; <i32> [#uses=1]
	br label %loopentry.1
loopentry.1:		; preds = %loopentry.1, %loopentry.1.outer
	%i.3 = phi i32 [ 0, %loopentry.1.outer ], [ %i.3.be, %loopentry.1 ]		; <i32> [#uses=2]
	%tmp.390 = load i32, ptr null		; <i32> [#uses=1]
	%tmp.392 = mul i32 %tmp.390, %j.2.1.ph		; <i32> [#uses=1]
	%tmp.394 = add i32 %tmp.392, %i.3		; <i32> [#uses=1]
	%i.3.be = add i32 %i.3, 1		; <i32> [#uses=1]
	br i1 false, label %loopentry.1, label %label.6
label.6:		; preds = %loopentry.1
	ret i32 %tmp.394
}

