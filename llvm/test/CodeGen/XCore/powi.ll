; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; RUN: grep "bl __powidf2" %t1.s | count 1
; RUN: grep "bl __powisf2" %t1.s | count 1
declare double @llvm.powi.f64(double, i32)

define double @test(double %F, i32 %power) {
        %result = call double @llvm.powi.f64(double %F, i32 %power)
	ret double %result
}

declare float @llvm.powi.f32(float, i32)

define float @testf(float %F, i32 %power) {
        %result = call float @llvm.powi.f32(float %F, i32 %power)
	ret float %result
}
