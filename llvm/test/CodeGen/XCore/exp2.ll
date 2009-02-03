; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; RUN: grep "bl exp2f" %t1.s | count 1
; RUN: grep "bl exp2" %t1.s | count 2
declare double @llvm.exp2.f64(double)

define double @test(double %F) {
        %result = call double @llvm.exp2.f64(double %F)
	ret double %result
}

declare float @llvm.exp2.f32(float)

define float @testf(float %F) {
        %result = call float @llvm.exp2.f32(float %F)
	ret float %result
}
