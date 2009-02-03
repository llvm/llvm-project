; RUN: llvm-as < %s | llc -march=x86 | grep cmp | count 1
; RUN: llvm-as < %s | llc -march=x86 | grep test | count 1

define i32 @f1(i32 %X, i32* %y) {
	%tmp = load i32* %y		; <i32> [#uses=1]
	%tmp.upgrd.1 = icmp eq i32 %tmp, 0		; <i1> [#uses=1]
	br i1 %tmp.upgrd.1, label %ReturnBlock, label %cond_true

cond_true:		; preds = %0
	ret i32 1

ReturnBlock:		; preds = %0
	ret i32 0
}

define i32 @f2(i32 %X, i32* %y) {
	%tmp = load i32* %y		; <i32> [#uses=1]
	%tmp1 = shl i32 %tmp, 3		; <i32> [#uses=1]
	%tmp1.upgrd.2 = icmp eq i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp1.upgrd.2, label %ReturnBlock, label %cond_true

cond_true:		; preds = %0
	ret i32 1

ReturnBlock:		; preds = %0
	ret i32 0
}
