; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; RUN: grep "bl powf" %t1.s | count 1
; RUN: grep "bl pow" %t1.s | count 2
declare double @llvm.pow.f64(double, double)

define double @test(double %F, double %power) {
        %result = call double @llvm.pow.f64(double %F, double %power)
	ret double %result
}

declare float @llvm.pow.f32(float, float)

define float @testf(float %F, float %power) {
        %result = call float @llvm.pow.f32(float %F, float %power)
	ret float %result
}
