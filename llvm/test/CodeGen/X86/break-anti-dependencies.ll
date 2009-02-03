; RUN: llvm-as < %s | llc -march=x86-64 -disable-post-RA-scheduler=false -break-anti-dependencies=false > %t
; RUN:   grep {%xmm0} %t | count 14
; RUN:   not grep {%xmm1} %t
; RUN: llvm-as < %s | llc -march=x86-64 -disable-post-RA-scheduler=false -break-anti-dependencies > %t
; RUN:   grep {%xmm0} %t | count 7
; RUN:   grep {%xmm1} %t | count 7

define void @goo(double* %r, double* %p, double* %q) nounwind {
entry:
	%0 = load double* %p, align 8
	%1 = add double %0, 1.100000e+00
	%2 = mul double %1, 1.200000e+00
	%3 = add double %2, 1.300000e+00
	%4 = mul double %3, 1.400000e+00
	%5 = add double %4, 1.500000e+00
	%6 = fptosi double %5 to i32
	%7 = load double* %r, align 8
	%8 = add double %7, 7.100000e+00
	%9 = mul double %8, 7.200000e+00
	%10 = add double %9, 7.300000e+00
	%11 = mul double %10, 7.400000e+00
	%12 = add double %11, 7.500000e+00
	%13 = fptosi double %12 to i32
	%14 = icmp slt i32 %6, %13
	br i1 %14, label %bb, label %return

bb:
	store double 9.300000e+00, double* %q, align 8
	ret void

return:
	ret void
}
