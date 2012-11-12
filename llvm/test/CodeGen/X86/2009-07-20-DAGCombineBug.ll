; RUN: llc < %s -march=x86

@bsBuff = internal global i32 0		; <i32*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 ()* @bsGetUInt32 to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define fastcc i32 @bsGetUInt32() nounwind ssp {
entry:
	%bsBuff.promoted44 = load i32* @bsBuff		; <i32> [#uses=1]
	%0 = add i32 0, -8		; <i32> [#uses=1]
	%1 = lshr i32 %bsBuff.promoted44, %0		; <i32> [#uses=1]
	%2 = shl i32 %1, 8		; <i32> [#uses=1]
	br label %bb3.i17

bb3.i9:		; preds = %bb3.i17
	br i1 false, label %bb2.i16, label %bb1.i15

bb1.i15:		; preds = %bb3.i9
	unreachable

bb2.i16:		; preds = %bb3.i9
	br label %bb3.i17

bb3.i17:		; preds = %bb2.i16, %entry
	br i1 false, label %bb3.i9, label %bsR.exit18

bsR.exit18:		; preds = %bb3.i17
	%3 = or i32 0, %2		; <i32> [#uses=0]
	ret i32 0
}
