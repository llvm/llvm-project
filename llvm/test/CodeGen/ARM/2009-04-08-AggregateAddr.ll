; RUN: llc -mtriple=arm-eabi %s -o /dev/null
; PR3795

define fastcc void @_D3foo3fooFAriZv({ i32, ptr } %d_arg, i32 %x_arg) {
entry:
	%d = alloca { i32, ptr }		; <ptr> [#uses=2]
	%x = alloca i32		; <ptr> [#uses=2]
	%b = alloca { double, double }		; <ptr> [#uses=1]
	store { i32, ptr } %d_arg, ptr %d
	store i32 %x_arg, ptr %x
	%tmp = load i32, ptr %x		; <i32> [#uses=1]
	%tmp1 = getelementptr { i32, ptr }, ptr %d, i32 0, i32 1		; <ptr> [#uses=1]
	%.ptr = load ptr, ptr %tmp1		; <ptr> [#uses=1]
	%tmp2 = getelementptr { double, double }, ptr %.ptr, i32 %tmp		; <ptr> [#uses=1]
	%tmp3 = load { double, double }, ptr %tmp2		; <{ double, double }> [#uses=1]
	store { double, double } %tmp3, ptr %b
	ret void
}
