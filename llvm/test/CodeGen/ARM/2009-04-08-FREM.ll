; RUN: llc < %s -march=arm

declare i32 @printf(i8*, ...)

define i32 @main() {
	%rem_r = frem double 0.000000e+00, 0.000000e+00		; <double> [#uses=1]
	%1 = call i32 (i8*, ...)* @printf(i8* null, double %rem_r)		; <i32> [#uses=0]
	ret i32 0
}
