; Without list-burr scheduling we may not see the difference in codegen here.
; Use a subtarget that has post-RA scheduling enabled because the anti-dependency
; breaker requires liveness information to be kept.
; RUN: llc < %s -march=x86-64 -mcpu=atom -enable-misched=false -post-RA-scheduler -pre-RA-sched=list-burr -break-anti-dependencies=none > %t
; RUN:   grep "%xmm0" %t | count 14
; RUN:   not grep "%xmm1" %t
; RUN: llc < %s -march=x86-64 -mcpu=atom -post-RA-scheduler -break-anti-dependencies=critical > %t
; RUN:   grep "%xmm0" %t | count 7
; RUN:   grep "%xmm1" %t | count 7

define void @goo(double* %r, double* %p, double* %q) nounwind {
entry:
	%0 = load double, double* %p, align 8
	%1 = fadd double %0, 1.100000e+00
	%2 = fmul double %1, 1.200000e+00
	%3 = fadd double %2, 1.300000e+00
	%4 = fmul double %3, 1.400000e+00
	%5 = fadd double %4, 1.500000e+00
	%6 = fptosi double %5 to i32
	%7 = load double, double* %r, align 8
	%8 = fadd double %7, 7.100000e+00
	%9 = fmul double %8, 7.200000e+00
	%10 = fadd double %9, 7.300000e+00
	%11 = fmul double %10, 7.400000e+00
	%12 = fadd double %11, 7.500000e+00
	%13 = fptosi double %12 to i32
	%14 = icmp slt i32 %6, %13
	br i1 %14, label %bb, label %return

bb:
	store double 9.300000e+00, double* %q, align 8
	ret void

return:
	ret void
}
