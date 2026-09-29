; RUN: llc -mtriple=arm-eabi %s -o /dev/null

declare i32 @printf(ptr, ...)

define i32 @main() {
	%rem_r = frem double 0.000000e+00, 0.000000e+00		; <double> [#uses=1]
	%1 = call i32 (ptr, ...) @printf(ptr null, double %rem_r)		; <i32> [#uses=0]
	ret i32 0
}
