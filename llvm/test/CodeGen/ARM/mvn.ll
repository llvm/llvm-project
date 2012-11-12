; RUN: llc < %s -march=arm | grep mvn | count 8

define i32 @f1() {
entry:
	ret i32 -1
}

define i32 @f2(i32 %a) {
entry:
	%tmpnot = xor i32 %a, -1		; <i32> [#uses=1]
	ret i32 %tmpnot
}

define i32 @f3(i32 %a) {
entry:
	%tmp1 = shl i32 %a, 2		; <i32> [#uses=1]
	%tmp1not = xor i32 %tmp1, -1		; <i32> [#uses=1]
	ret i32 %tmp1not
}

define i32 @f4(i32 %a, i8 %b) {
entry:
	%shift.upgrd.1 = zext i8 %b to i32		; <i32> [#uses=1]
	%tmp3 = shl i32 %a, %shift.upgrd.1		; <i32> [#uses=1]
	%tmp3not = xor i32 %tmp3, -1		; <i32> [#uses=1]
	ret i32 %tmp3not
}

define i32 @f5(i32 %a) {
entry:
	%tmp1 = lshr i32 %a, 2		; <i32> [#uses=1]
	%tmp1not = xor i32 %tmp1, -1		; <i32> [#uses=1]
	ret i32 %tmp1not
}

define i32 @f6(i32 %a, i8 %b) {
entry:
	%shift.upgrd.2 = zext i8 %b to i32		; <i32> [#uses=1]
	%tmp2 = lshr i32 %a, %shift.upgrd.2		; <i32> [#uses=1]
	%tmp2not = xor i32 %tmp2, -1		; <i32> [#uses=1]
	ret i32 %tmp2not
}

define i32 @f7(i32 %a) {
entry:
	%tmp1 = ashr i32 %a, 2		; <i32> [#uses=1]
	%tmp1not = xor i32 %tmp1, -1		; <i32> [#uses=1]
	ret i32 %tmp1not
}

define i32 @f8(i32 %a, i8 %b) {
entry:
	%shift.upgrd.3 = zext i8 %b to i32		; <i32> [#uses=1]
	%tmp3 = ashr i32 %a, %shift.upgrd.3		; <i32> [#uses=1]
	%tmp3not = xor i32 %tmp3, -1		; <i32> [#uses=1]
	ret i32 %tmp3not
}

define i32 @f9() {
entry:
	%tmp4845 = add i32 0, 0		; <i32> [#uses=1]
	br label %cond_true4848

cond_true4848:		; preds = %entry
	%tmp4851 = sub i32 -3, 0		; <i32> [#uses=1]
	%abc = add i32 %tmp4851, %tmp4845		; <i32> [#uses=1]
	ret i32 %abc
}

define i1 @f10(i32 %a) {
entry:
	%tmp102 = icmp eq i32 -2, %a		; <i1> [#uses=1]
	ret i1 %tmp102
}
