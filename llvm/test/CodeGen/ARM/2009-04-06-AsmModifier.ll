; RUN: llc -mtriple=arm-eabi -no-integrated-as %s -o - | FileCheck %s

define i32 @_swilseek(i32) nounwind {
entry:
	%ptr = alloca i32		; <ptr> [#uses=2]
	store i32 %0, ptr %ptr
	%retval = alloca i32		; <ptr> [#uses=2]
	store i32 0, ptr %retval
	%res = alloca i32		; <ptr> [#uses=0]
	%fh = alloca i32		; <ptr> [#uses=1]
	%1 = load i32, ptr %fh		; <i32> [#uses=1]
	%2 = load i32, ptr %ptr		; <i32> [#uses=1]
	%3 = call i32 asm "mov r0, $2; mov r1, $3; swi ${1:a}; mov $0, r0", "=r,i,r,r,~{r0},~{r1}"(i32 107, i32 %1, i32 %2) nounwind		; <i32> [#uses=1]
        store i32 %3, ptr %retval
	br label %return

return:		; preds = %entry
	%4 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %4
}

; CHECK: swi 107

