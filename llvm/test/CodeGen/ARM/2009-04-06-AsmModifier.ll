; RUN: llc < %s -march=arm | grep "swi 107"

define i32 @_swilseek(i32) nounwind {
entry:
	%ptr = alloca i32		; <i32*> [#uses=2]
	store i32 %0, i32* %ptr
	%retval = alloca i32		; <i32*> [#uses=2]
	store i32 0, i32* %retval
	%res = alloca i32		; <i32*> [#uses=0]
	%fh = alloca i32		; <i32*> [#uses=1]
	%1 = load i32* %fh		; <i32> [#uses=1]
	%2 = load i32* %ptr		; <i32> [#uses=1]
	%3 = call i32 asm "mov r0, $2; mov r1, $3; swi ${1:a}; mov $0, r0", "=r,i,r,r,~{r0},~{r1}"(i32 107, i32 %1, i32 %2) nounwind		; <i32> [#uses=1]
        store i32 %3, i32* %retval
	br label %return

return:		; preds = %entry
	%4 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %4
}
