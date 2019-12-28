; RUN: llc -verify-machineinstrs -mattr=-vsx \
; RUN:   -mattr=+allow-unaligned-fp-access < %s | FileCheck %s
; ModuleID = 'foo.c'

target triple = "powerpc-unknown-linux-gnu"

	%struct.anon = type <{ i8, float }>
@s = global %struct.anon <{ i8 3, float 0x4014666660000000 }>		; <%struct.anon*> [#uses=1]
@u = global <{ i8, double }> <{ i8 3, double 5.100000e+00 }>		; <<{ i8, double }>*> [#uses=1]
@t = weak global %struct.anon zeroinitializer		; <%struct.anon*> [#uses=2]
@v = weak global <{ i8, double }> zeroinitializer		; <<{ i8, double }>*> [#uses=2]
@.str = internal constant [8 x i8] c"%f %lf\0A\00"		; <[8 x i8]*> [#uses=1]

; CHECK: foo
; CHECK: lfs
; CHECK: lfd
; CHECK: stfs
; CHECK: stfd
; CHECK: blr
define i32 @foo() {
entry:
	%retval = alloca i32, align 4		; <i32*> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = getelementptr %struct.anon, %struct.anon* @s, i32 0, i32 1		; <float*> [#uses=1]
	%tmp1 = load float, float* %tmp, align 1		; <float> [#uses=1]
	%tmp2 = getelementptr %struct.anon, %struct.anon* @t, i32 0, i32 1		; <float*> [#uses=1]
	store float %tmp1, float* %tmp2, align 1
	%tmp3 = getelementptr <{ i8, double }>, <{ i8, double }>* @u, i32 0, i32 1		; <double*> [#uses=1]
	%tmp4 = load double, double* %tmp3, align 1		; <double> [#uses=1]
	%tmp5 = getelementptr <{ i8, double }>, <{ i8, double }>* @v, i32 0, i32 1		; <double*> [#uses=1]
	store double %tmp4, double* %tmp5, align 1
	br label %return

return:		; preds = %entry
	%retval6 = load i32, i32* %retval		; <i32> [#uses=1]
	ret i32 %retval6
}

; CHECK: main
; CHECK: lfs
; CHECK: lfd
; CHECK: blr
define i32 @main() {
entry:
	%retval = alloca i32, align 4		; <i32*> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = call i32 @foo( )		; <i32> [#uses=0]
	%tmp1 = getelementptr %struct.anon, %struct.anon* @t, i32 0, i32 1		; <float*> [#uses=1]
	%tmp2 = load float, float* %tmp1, align 1		; <float> [#uses=1]
	%tmp23 = fpext float %tmp2 to double		; <double> [#uses=1]
	%tmp4 = getelementptr <{ i8, double }>, <{ i8, double }>* @v, i32 0, i32 1		; <double*> [#uses=1]
	%tmp5 = load double, double* %tmp4, align 1		; <double> [#uses=1]
	%tmp6 = getelementptr [8 x i8], [8 x i8]* @.str, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp7 = call i32 (i8*, ...) @printf( i8* %tmp6, double %tmp23, double %tmp5 )		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	%retval8 = load i32, i32* %retval		; <i32> [#uses=1]
	ret i32 %retval8
}

declare i32 @printf(i8*, ...)
