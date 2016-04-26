; RUN: opt < %s -globalopt -S | FileCheck %s

; The 'X' indices could be larger than 31.  Do not SROA the outer
; indices of this array.
; CHECK: @mm = {{.*}} [16 x [31 x double]] zeroinitializer
@mm = internal global [16 x [31 x double]] zeroinitializer, align 32

define void @test(i32 %X) {
	%P = getelementptr [16 x [31 x double]], [16 x [31 x double]]* @mm, i32 0, i32 0, i32 %X
	store double 1.0, double* %P
	ret void
}

define double @get(i32 %X) {
	%P = getelementptr [16 x [31 x double]], [16 x [31 x double]]* @mm, i32 0, i32 0, i32 %X
	%V = load double, double* %P
	ret double %V
}
