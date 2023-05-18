; RUN: llc < %s -mtriple=arm-apple-ios -mattr=+vfp2 -disable-post-ra | FileCheck %s
; RUN: llc < %s -mtriple=arm-apple-ios -mattr=+vfp2 -disable-post-ra -regalloc=basic | FileCheck %s

define void @test(ptr %P, ptr %D) {
	%A = load float, ptr %P		; <float> [#uses=1]
	%B = load double, ptr %D		; <double> [#uses=1]
	store float %A, ptr %P
	store double %B, ptr %D
	ret void
}

declare float @fabsf(float)

declare double @fabs(double)

define void @test_abs(ptr %P, ptr %D) {
;CHECK-LABEL: test_abs:
	%a = load float, ptr %P		; <float> [#uses=1]
;CHECK: vabs.f32
	%b = call float @fabsf( float %a ) readnone	; <float> [#uses=1]
	store float %b, ptr %P
	%A = load double, ptr %D		; <double> [#uses=1]
;CHECK: vabs.f64
	%B = call double @fabs( double %A ) readnone	; <double> [#uses=1]
	store double %B, ptr %D
	ret void
}

define void @test_add(ptr %P, ptr %D) {
;CHECK-LABEL: test_add:
	%a = load float, ptr %P		; <float> [#uses=2]
	%b = fadd float %a, %a		; <float> [#uses=1]
	store float %b, ptr %P
	%A = load double, ptr %D		; <double> [#uses=2]
	%B = fadd double %A, %A		; <double> [#uses=1]
	store double %B, ptr %D
	ret void
}

define void @test_ext_round(ptr %P, ptr %D) {
;CHECK-LABEL: test_ext_round:
	%a = load float, ptr %P		; <float> [#uses=1]
;CHECK-DAG: vcvt.f64.f32
;CHECK-DAG: vcvt.f32.f64
	%b = fpext float %a to double		; <double> [#uses=1]
	%A = load double, ptr %D		; <double> [#uses=1]
	%B = fptrunc double %A to float		; <float> [#uses=1]
	store double %b, ptr %D
	store float %B, ptr %P
	ret void
}

define void @test_fma(ptr %P1, ptr %P2, ptr %P3) {
;CHECK-LABEL: test_fma:
	%a1 = load float, ptr %P1		; <float> [#uses=1]
	%a2 = load float, ptr %P2		; <float> [#uses=1]
	%a3 = load float, ptr %P3		; <float> [#uses=1]
;CHECK: vnmls.f32
	%X = fmul float %a1, %a2		; <float> [#uses=1]
	%Y = fsub float %X, %a3		; <float> [#uses=1]
	store float %Y, ptr %P1
	ret void
}

define i32 @test_ftoi(ptr %P1) {
;CHECK-LABEL: test_ftoi:
	%a1 = load float, ptr %P1		; <float> [#uses=1]
;CHECK: vcvt.s32.f32
	%b1 = fptosi float %a1 to i32		; <i32> [#uses=1]
	ret i32 %b1
}

define i32 @test_ftou(ptr %P1) {
;CHECK-LABEL: test_ftou:
	%a1 = load float, ptr %P1		; <float> [#uses=1]
;CHECK: vcvt.u32.f32
	%b1 = fptoui float %a1 to i32		; <i32> [#uses=1]
	ret i32 %b1
}

define i32 @test_dtoi(ptr %P1) {
;CHECK-LABEL: test_dtoi:
	%a1 = load double, ptr %P1		; <double> [#uses=1]
;CHECK: vcvt.s32.f64
	%b1 = fptosi double %a1 to i32		; <i32> [#uses=1]
	ret i32 %b1
}

define i32 @test_dtou(ptr %P1) {
;CHECK-LABEL: test_dtou:
	%a1 = load double, ptr %P1		; <double> [#uses=1]
;CHECK: vcvt.u32.f64
	%b1 = fptoui double %a1 to i32		; <i32> [#uses=1]
	ret i32 %b1
}

define void @test_utod(ptr %P1, i32 %X) {
;CHECK-LABEL: test_utod:
;CHECK: vcvt.f64.u32
	%b1 = uitofp i32 %X to double		; <double> [#uses=1]
	store double %b1, ptr %P1
	ret void
}

define void @test_utod2(ptr %P1, i8 %X) {
;CHECK-LABEL: test_utod2:
;CHECK: vcvt.f64.u32
	%b1 = uitofp i8 %X to double		; <double> [#uses=1]
	store double %b1, ptr %P1
	ret void
}

define void @test_cmp(ptr %glob, i32 %X) {
;CHECK-LABEL: test_cmp:
entry:
	%tmp = load float, ptr %glob		; <float> [#uses=2]
	%tmp3 = getelementptr float, ptr %glob, i32 2		; <ptr> [#uses=1]
	%tmp4 = load float, ptr %tmp3		; <float> [#uses=2]
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

define void @test_cmpfp0(ptr %glob, i32 %X) {
;CHECK-LABEL: test_cmpfp0:
entry:
	%tmp = load float, ptr %glob		; <float> [#uses=1]
;CHECK: vcmp.f32
	%tmp.upgrd.3 = fcmp ogt float %tmp, 0.000000e+00		; <i1> [#uses=1]
	br i1 %tmp.upgrd.3, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	%tmp.upgrd.4 = tail call i32 (...) @bar( )		; <i32> [#uses=0]
	ret void

cond_false:		; preds = %entry
	%tmp1 = tail call i32 (...) @baz( )		; <i32> [#uses=0]
	ret void
}
