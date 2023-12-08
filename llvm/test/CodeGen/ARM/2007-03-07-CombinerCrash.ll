; RUN: llc < %s -mtriple=arm-apple-darwin -mattr=+v6,+vfp2

define fastcc ptr @read_sleb128(ptr %p, ptr %val) {
	br label %bb

bb:		; preds = %bb, %0
	%p_addr.0 = getelementptr i8, ptr %p, i32 0		; <ptr> [#uses=1]
	%tmp2 = load i8, ptr %p_addr.0		; <i8> [#uses=2]
	%tmp4.rec = add i32 0, 1		; <i32> [#uses=1]
	%tmp4 = getelementptr i8, ptr %p, i32 %tmp4.rec		; <ptr> [#uses=1]
	%tmp56 = zext i8 %tmp2 to i32		; <i32> [#uses=1]
	%tmp7 = and i32 %tmp56, 127		; <i32> [#uses=1]
	%tmp9 = shl i32 %tmp7, 0		; <i32> [#uses=1]
	%tmp11 = or i32 %tmp9, 0		; <i32> [#uses=1]
	icmp slt i8 %tmp2, 0		; <i1>:1 [#uses=1]
	br i1 %1, label %bb, label %cond_next28

cond_next28:		; preds = %bb
	store i32 %tmp11, ptr %val
	ret ptr %tmp4
}
