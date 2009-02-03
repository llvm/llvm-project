; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; RUN: grep "bl log2f" %t1.s | count 1
; RUN: grep "bl log2" %t1.s | count 2
declare double @llvm.log2.f64(double)

define double @test(double %F) {
        %result = call double @llvm.log2.f64(double %F)
	ret double %result
}

declare float @llvm.log2.f32(float)

define float @testf(float %F) {
        %result = call float @llvm.log2.f32(float %F)
	ret float %result
}
