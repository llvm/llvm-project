; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin -mattr=+v6,+vfp2

define fastcc i8* @read_sleb128(i8* %p, i32* %val) {
	br label %bb

bb:		; preds = %bb, %0
	%p_addr.0 = getelementptr i8* %p, i32 0		; <i8*> [#uses=1]
	%tmp2 = load i8* %p_addr.0		; <i8> [#uses=2]
	%tmp4.rec = add i32 0, 1		; <i32> [#uses=1]
	%tmp4 = getelementptr i8* %p, i32 %tmp4.rec		; <i8*> [#uses=1]
	%tmp56 = zext i8 %tmp2 to i32		; <i32> [#uses=1]
	%tmp7 = and i32 %tmp56, 127		; <i32> [#uses=1]
	%tmp9 = shl i32 %tmp7, 0		; <i32> [#uses=1]
	%tmp11 = or i32 %tmp9, 0		; <i32> [#uses=1]
	icmp slt i8 %tmp2, 0		; <i1>:1 [#uses=1]
	br i1 %1, label %bb, label %cond_next28

cond_next28:		; preds = %bb
	store i32 %tmp11, i32* %val
	ret i8* %tmp4
}
