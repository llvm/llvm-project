; RUN: llvm-as < %s | llc -march=arm -mtriple=arm-linux-gnueabi | \
; RUN:   grep {add sp, sp, #16} | count 1
; RUN: llvm-as < %s | llc -march=arm -mtriple=arm-linux-gnu | \
; RUN:   grep {add sp, sp, #12} | count 2

define i32 @f(i32 %a, ...) {
entry:
	%a_addr = alloca i32		; <i32*> [#uses=1]
	%retval = alloca i32, align 4		; <i32*> [#uses=2]
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %a, i32* %a_addr
	store i32 0, i32* %tmp
	%tmp1 = load i32* %tmp		; <i32> [#uses=1]
	store i32 %tmp1, i32* %retval
	br label %return

return:		; preds = %entry
	%retval2 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval2
}
