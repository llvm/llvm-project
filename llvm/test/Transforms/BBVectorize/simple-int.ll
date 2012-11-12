target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s

declare double @llvm.fma.f64(double, double, double)
declare double @llvm.cos.f64(double)
declare double @llvm.powi.f64(double, i32)

; Basic depth-3 chain with fma
define double @test1(double %A1, double %A2, double %B1, double %B2, double %C1, double %C2) {
	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
	%Y1 = call double @llvm.fma.f64(double %X1, double %A1, double %C1)
	%Y2 = call double @llvm.fma.f64(double %X2, double %A2, double %C2)
	%Z1 = fadd double %Y1, %B1
	%Z2 = fadd double %Y2, %B2
	%R  = fmul double %Z1, %Z2
	ret double %R
; CHECK: @test1
; CHECK: %X1.v.i1.1 = insertelement <2 x double> undef, double %B1, i32 0
; CHECK: %X1.v.i1.2 = insertelement <2 x double> %X1.v.i1.1, double %B2, i32 1
; CHECK: %X1.v.i0.1 = insertelement <2 x double> undef, double %A1, i32 0
; CHECK: %X1.v.i0.2 = insertelement <2 x double> %X1.v.i0.1, double %A2, i32 1
; CHECK: %X1 = fsub <2 x double> %X1.v.i0.2, %X1.v.i1.2
; CHECK: %Y1.v.i2.1 = insertelement <2 x double> undef, double %C1, i32 0
; CHECK: %Y1.v.i2.2 = insertelement <2 x double> %Y1.v.i2.1, double %C2, i32 1
; CHECK: %Y1 = call <2 x double> @llvm.fma.v2f64(<2 x double> %X1, <2 x double> %X1.v.i0.2, <2 x double> %Y1.v.i2.2)
; CHECK: %Z1 = fadd <2 x double> %Y1, %X1.v.i1.2
; CHECK: %Z1.v.r1 = extractelement <2 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <2 x double> %Z1, i32 1
; CHECK: %R = fmul double %Z1.v.r1, %Z1.v.r2
; CHECK: ret double %R
}

; Basic depth-3 chain with cos
define double @test2(double %A1, double %A2, double %B1, double %B2) {
	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
	%Y1 = call double @llvm.cos.f64(double %X1)
	%Y2 = call double @llvm.cos.f64(double %X2)
	%Z1 = fadd double %Y1, %B1
	%Z2 = fadd double %Y2, %B2
	%R  = fmul double %Z1, %Z2
	ret double %R
; CHECK: @test2
; CHECK: %X1.v.i1.1 = insertelement <2 x double> undef, double %B1, i32 0
; CHECK: %X1.v.i1.2 = insertelement <2 x double> %X1.v.i1.1, double %B2, i32 1
; CHECK: %X1.v.i0.1 = insertelement <2 x double> undef, double %A1, i32 0
; CHECK: %X1.v.i0.2 = insertelement <2 x double> %X1.v.i0.1, double %A2, i32 1
; CHECK: %X1 = fsub <2 x double> %X1.v.i0.2, %X1.v.i1.2
; CHECK: %Y1 = call <2 x double> @llvm.cos.v2f64(<2 x double> %X1)
; CHECK: %Z1 = fadd <2 x double> %Y1, %X1.v.i1.2
; CHECK: %Z1.v.r1 = extractelement <2 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <2 x double> %Z1, i32 1
; CHECK: %R = fmul double %Z1.v.r1, %Z1.v.r2
; CHECK: ret double %R
}

; Basic depth-3 chain with powi
define double @test3(double %A1, double %A2, double %B1, double %B2, i32 %P) {

	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
	%Y1 = call double @llvm.powi.f64(double %X1, i32 %P)
	%Y2 = call double @llvm.powi.f64(double %X2, i32 %P)
	%Z1 = fadd double %Y1, %B1
	%Z2 = fadd double %Y2, %B2
	%R  = fmul double %Z1, %Z2
	ret double %R
; CHECK: @test3
; CHECK: %X1.v.i1.1 = insertelement <2 x double> undef, double %B1, i32 0
; CHECK: %X1.v.i1.2 = insertelement <2 x double> %X1.v.i1.1, double %B2, i32 1
; CHECK: %X1.v.i0.1 = insertelement <2 x double> undef, double %A1, i32 0
; CHECK: %X1.v.i0.2 = insertelement <2 x double> %X1.v.i0.1, double %A2, i32 1
; CHECK: %X1 = fsub <2 x double> %X1.v.i0.2, %X1.v.i1.2
; CHECK: %Y1 = call <2 x double> @llvm.powi.v2f64(<2 x double> %X1, i32 %P)
; CHECK: %Z1 = fadd <2 x double> %Y1, %X1.v.i1.2
; CHECK: %Z1.v.r1 = extractelement <2 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <2 x double> %Z1, i32 1
; CHECK: %R = fmul double %Z1.v.r1, %Z1.v.r2
; CHECK: ret double %R
}

; Basic depth-3 chain with powi (different powers: should not vectorize)
define double @test4(double %A1, double %A2, double %B1, double %B2, i32 %P) {

	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
        %P2 = add i32 %P, 1
	%Y1 = call double @llvm.powi.f64(double %X1, i32 %P)
	%Y2 = call double @llvm.powi.f64(double %X2, i32 %P2)
	%Z1 = fadd double %Y1, %B1
	%Z2 = fadd double %Y2, %B2
	%R  = fmul double %Z1, %Z2
	ret double %R
; CHECK: @test4
; CHECK-NOT: <2 x double>
; CHECK: ret double %R
}

; CHECK: declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
; CHECK: declare <2 x double> @llvm.cos.v2f64(<2 x double>) nounwind readonly
; CHECK: declare <2 x double> @llvm.powi.v2f64(<2 x double>, i32) nounwind readonly

