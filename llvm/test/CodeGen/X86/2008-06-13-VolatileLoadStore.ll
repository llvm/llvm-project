; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movsd | count 5
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movl | count 2

@atomic = global double 0.000000e+00		; <double*> [#uses=1]
@atomic2 = global double 0.000000e+00		; <double*> [#uses=1]
@anything = global i64 0		; <i64*> [#uses=1]
@ioport = global i32 0		; <i32*> [#uses=2]

define i16 @f(i64 %x, double %y) {
	%b = bitcast i64 %x to double		; <double> [#uses=1]
	volatile store double %b, double* @atomic ; one processor operation only
	volatile store double 0.000000e+00, double* @atomic2 ; one processor operation only
	%b2 = bitcast double %y to i64		; <i64> [#uses=1]
	volatile store i64 %b2, i64* @anything ; may transform to store of double
	%l = volatile load i32* @ioport		; must not narrow
	%t = trunc i32 %l to i16		; <i16> [#uses=1]
	%l2 = volatile load i32* @ioport		; must not narrow
	%tmp = lshr i32 %l2, 16		; <i32> [#uses=1]
	%t2 = trunc i32 %tmp to i16		; <i16> [#uses=1]
	%f = add i16 %t, %t2		; <i16> [#uses=1]
	ret i16 %f
}
