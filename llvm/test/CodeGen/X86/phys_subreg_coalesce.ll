; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin9 -mattr=+sse2 | not grep movl

	%struct.dpoint = type { double, double }

define %struct.dpoint @midpoint(i64 %p1.0, i64 %p2.0) nounwind readnone {
entry:
	%0 = trunc i64 %p1.0 to i32		; <i32> [#uses=1]
	%1 = sitofp i32 %0 to double		; <double> [#uses=1]
	%2 = trunc i64 %p2.0 to i32		; <i32> [#uses=1]
	%3 = sitofp i32 %2 to double		; <double> [#uses=1]
	%4 = add double %1, %3		; <double> [#uses=1]
	%5 = mul double %4, 5.000000e-01		; <double> [#uses=1]
	%6 = lshr i64 %p1.0, 32		; <i64> [#uses=1]
	%7 = trunc i64 %6 to i32		; <i32> [#uses=1]
	%8 = sitofp i32 %7 to double		; <double> [#uses=1]
	%9 = lshr i64 %p2.0, 32		; <i64> [#uses=1]
	%10 = trunc i64 %9 to i32		; <i32> [#uses=1]
	%11 = sitofp i32 %10 to double		; <double> [#uses=1]
	%12 = add double %8, %11		; <double> [#uses=1]
	%13 = mul double %12, 5.000000e-01		; <double> [#uses=1]
	%mrv3 = insertvalue %struct.dpoint undef, double %5, 0		; <%struct.dpoint> [#uses=1]
	%mrv4 = insertvalue %struct.dpoint %mrv3, double %13, 1		; <%struct.dpoint> [#uses=1]
	ret %struct.dpoint %mrv4
}
