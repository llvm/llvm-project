; RUN: llc < %s -mtriple=arm-apple-darwin -mattr=+vfp2

	%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ptr, i32 }
@"\01LC10" = external constant [54 x i8]		; <ptr> [#uses=1]

define fastcc void @t() {
entry:
	%0 = load i64, ptr null, align 4		; <i64> [#uses=1]
	%1 = uitofp i64 %0 to double		; <double> [#uses=1]
	%2 = fdiv double 0.000000e+00, %1		; <double> [#uses=1]
	%3 = call i32 (ptr, ptr, ...) @fprintf(ptr null, ptr @"\01LC10", i64 0, double %2)		; <i32> [#uses=0]
	ret void
}

declare i32 @fprintf(ptr, ptr, ...)
