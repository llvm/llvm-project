; RUN: llvm-as < %s | llc -march=arm | grep bic | count 2
; RUN: llvm-as < %s | llc -march=arm -mattr=+v6,+vfp2 | \
; RUN:   grep fneg | count 2

define float @test1(float %x, double %y) {
	%tmp = fpext float %x to double
	%tmp2 = tail call double @copysign( double %tmp, double %y )
	%tmp3 = fptrunc double %tmp2 to float
	ret float %tmp3
}

define double @test2(double %x, float %y) {
	%tmp = fpext float %y to double
	%tmp2 = tail call double @copysign( double %x, double %tmp )
	ret double %tmp2
}

declare double @copysign(double, double)
