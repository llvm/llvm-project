; RUN: llvm-as < %s | opt -globalopt | llvm-dis | grep {16 x .31 x double.. zeroinitializer}

; The 'X' indices could be larger than 31.  Do not SROA the outer indices of this array.
@mm = internal global [16 x [31 x double]] zeroinitializer, align 32

define void @test(i32 %X) {
	%P = getelementptr [16 x [31 x double]]* @mm, i32 0, i32 0, i32 %X
	store double 1.0, double* %P
	ret void
}

define double @get(i32 %X) {
	%P = getelementptr [16 x [31 x double]]* @mm, i32 0, i32 0, i32 %X
	%V = load double* %P
	ret double %V
}
