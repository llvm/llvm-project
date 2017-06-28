; RUN: llc < %s -mtriple=arm-apple-ios -mattr=+vfp2 -disable-post-ra | FileCheck %s
; RUN: llc < %s -mtriple=arm-apple-ios -mattr=+vfp2 -disable-post-ra -regalloc=basic | FileCheck %s

define void @test(float* %P, double* %D) {
	%A = load float, float* %P		; <float> [#uses=1]
	%B = load double, double* %D		; <double> [#uses=1]
	store float %A, float* %P
	store double %B, double* %D
	ret void
}

declare float @fabsf(float)

declare double @fabs(double)

define void @test_abs(float* %P, double* %D) {
;CHECK-LABEL: test_abs:
	%a = load float, float* %P		; <float> [#uses=1]
;CHECK: vabs.f32
	%b = call float @fabsf( float %a ) readnone	; <float> [#uses=1]
	store float %b, float* %P
	%A = load double, double* %D		; <double> [#uses=1]
;CHECK: vabs.f64
	%B = call double @fabs( double %A ) readnone	; <double> [#uses=1]
	store double %B, double* %D
	ret void
}

define void @test_add(float* %P, double* %D) {
;CHECK-LABEL: test_add:
	%a = load float, float* %P		; <float> [#uses=2]
	%b = fadd float %a, %a		; <float> [#uses=1]
	store float %b, float* %P
	%A = load double, double* %D		; <double> [#uses=2]
	%B = fadd double %A, %A		; <double> [#uses=1]
	store double %B, double* %D
	ret void
}

define void @test_ext_round(float* %P, double* %D) {
;CHECK-LABEL: test_ext_round:
	%a = load float, float* %P		; <float> [#uses=1]
;CHECK-DAG: vcvt.f64.f32
;CHECK-DAG: vcvt.f32.f64
	%b = fpext float %a to double		; <double> [#uses=1]
	%A = load double, double* %D		; <double> [#uses=1]
	%B = fptrunc double %A to float		; <float> [#uses=1]
	store double %b, double* %D
	store float %B, float* %P
	ret void
}

define void @test_fma(float* %P1, float* %P2, float* %P3) {
;CHECK-LABEL: test_fma:
	%a1 = load float, float* %P1		; <float> [#uses=1]
	%a2 = load float, float* %P2		; <float> [#uses=1]
	%a3 = load float, float* %P3		; <float> [#uses=1]
;CHECK: vnmls.f32
	%X = fmul float %a1, %a2		; <float> [#uses=1]
	%Y = fsub float %X, %a3		; <float> [#uses=1]
	store float %Y, float* %P1
	ret void
}

define i32 @test_ftoi(float* %P1) {
;CHECK-LABEL: test_ftoi:
	%a1 = load float, float* %P1		; <float> [#uses=1]
;CHECK: vcvt.s32.f32
	%b1 = fptosi float %a1 to i32		; <i32> [#uses=1]
	ret i32 %b1
}

define i32 @test_ftou(float* %P1) {
;CHECK-LABEL: test_ftou:
	%a1 = load float, float* %P1		; <float> [#uses=1]
;CHECK: vcvt.u32.f32
	%b1 = fptoui float %a1 to i32		; <i32> [#uses=1]
	ret i32 %b1
}

define i32 @test_dtoi(double* %P1) {
;CHECK-LABEL: test_dtoi:
	%a1 = load double, double* %P1		; <double> [#uses=1]
;CHECK: vcvt.s32.f64
	%b1 = fptosi double %a1 to i32		; <i32> [#uses=1]
	ret i32 %b1
}

define i32 @test_dtou(double* %P1) {
;CHECK-LABEL: test_dtou:
	%a1 = load double, double* %P1		; <double> [#uses=1]
;CHECK: vcvt.u32.f64
	%b1 = fptoui double %a1 to i32		; <i32> [#uses=1]
	ret i32 %b1
}

define void @test_utod(double* %P1, i32 %X) {
;CHECK-LABEL: test_utod:
;CHECK: vcvt.f64.u32
	%b1 = uitofp i32 %X to double		; <double> [#uses=1]
	store double %b1, double* %P1
	ret void
}

define void @test_utod2(double* %P1, i8 %X) {
;CHECK-LABEL: test_utod2:
;CHECK: vcvt.f64.u32
	%b1 = uitofp i8 %X to double		; <double> [#uses=1]
	store double %b1, double* %P1
	ret void
}

define void @test_cmp(float* %glob, i32 %X) {
;CHECK-LABEL: test_cmp:
entry:
	%tmp = load float, float* %glob		; <float> [#uses=2]
	%tmp3 = getelementptr float, float* %glob, i32 2		; <float*> [#uses=1]
	%tmp4 = load float, float* %tmp3		; <float> [#uses=2]
	%tmp.upgrd.1 = fcmp oeq float %tmp, %tmp4		; <i1> [#uses=1]
	%tmp5 = fcmp uno float %tmp, %tmp4		; <i1> [#uses=1]
	%tmp6 = or i1 %tmp.upgrd.1, %tmp5		; <i1> [#uses=1]
;CHECK: bmi
;CHECK-NEXT: bgt
	br i1 %tmp6, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	%tmp.upgrd.2 = tail call i32 (...) @bar( )		; <i32> [#uses=0]
	ret void

cond_false:		; preds = %entry
	%tmp7 = tail call i32 (...) @baz( )		; <i32> [#uses=0]
	ret void
}

declare i1 @llvm.isunordered.f32(float, float)

declare i32 @bar(...)

declare i32 @baz(...)

define void @test_cmpfp0(float* %glob, i32 %X) {
;CHECK-LABEL: test_cmpfp0:
entry:
	%tmp = load float, float* %glob		; <float> [#uses=1]
;CHECK: vcmpe.f32
	%tmp.upgrd.3 = fcmp ogt float %tmp, 0.000000e+00		; <i1> [#uses=1]
	br i1 %tmp.upgrd.3, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	%tmp.upgrd.4 = tail call i32 (...) @bar( )		; <i32> [#uses=0]
	ret void

cond_false:		; preds = %entry
	%tmp1 = tail call i32 (...) @baz( )		; <i32> [#uses=0]
	ret void
}
