; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; RUN: grep "bl sinf" %t1.s | count 1
; RUN: grep "bl sin" %t1.s | count 2
declare double @llvm.sin.f64(double)

define double @test(double %F) {
        %result = call double @llvm.sin.f64(double %F)
	ret double %result
}

declare float @llvm.sin.f32(float)

define float @testf(float %F) {
        %result = call float @llvm.sin.f32(float %F)
	ret float %result
}
