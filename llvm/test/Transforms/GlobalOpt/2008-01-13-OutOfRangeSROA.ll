; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; The 'X' indices could be larger than 31.  Do not SROA the outer
; indices of this array.
; CHECK: @mm = {{.*}} [16 x [31 x double]] zeroinitializer
@mm = internal global [16 x [31 x double]] zeroinitializer, align 32

define void @test(i32 %X) {
	%P = getelementptr [16 x [31 x double]], ptr @mm, i32 0, i32 0, i32 %X
	store double 1.0, ptr %P
	ret void
}

define double @get(i32 %X) {
	%P = getelementptr [16 x [31 x double]], ptr @mm, i32 0, i32 0, i32 %X
	%V = load double, ptr %P
	ret double %V
}
