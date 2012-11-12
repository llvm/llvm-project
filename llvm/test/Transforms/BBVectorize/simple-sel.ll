target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -bb-vectorize-no-bools -instcombine -gvn -S | FileCheck %s -check-prefix=CHECK-NB

; Basic depth-3 chain with select
define double @test1(double %A1, double %A2, double %B1, double %B2, i1 %C1, i1 %C2) {
; CHECK: @test1
; CHECK: %X1.v.i1.1 = insertelement <2 x double> undef, double %B1, i32 0
; CHECK: %X1.v.i1.2 = insertelement <2 x double> %X1.v.i1.1, double %B2, i32 1
; CHECK: %X1.v.i0.1 = insertelement <2 x double> undef, double %A1, i32 0
; CHECK: %X1.v.i0.2 = insertelement <2 x double> %X1.v.i0.1, double %A2, i32 1
	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
; CHECK: %X1 = fsub <2 x double> %X1.v.i0.2, %X1.v.i1.2
	%Y1 = fmul double %X1, %A1
	%Y2 = fmul double %X2, %A2
; CHECK: %Y1 = fmul <2 x double> %X1, %X1.v.i0.2
        %Z1 = select i1 %C1, double %Y1, double %B1
        %Z2 = select i1 %C2, double %Y2, double %B2
; CHECK: %Z1.v.i0.1 = insertelement <2 x i1> undef, i1 %C1, i32 0
; CHECK: %Z1.v.i0.2 = insertelement <2 x i1> %Z1.v.i0.1, i1 %C2, i32 1
; CHECK: %Z1 = select <2 x i1> %Z1.v.i0.2, <2 x double> %Y1, <2 x double> %X1.v.i1.2
	%R  = fmul double %Z1, %Z2
; CHECK: %Z1.v.r1 = extractelement <2 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <2 x double> %Z1, i32 1
; CHECK: %R = fmul double %Z1.v.r1, %Z1.v.r2
	ret double %R
; CHECK: ret double %R
}

; Basic depth-3 chain with select (and vect. compare)
define double @test2(double %A1, double %A2, double %B1, double %B2) {
; CHECK: @test2
; CHECK-NB: @test2
; CHECK: %X1.v.i1.1 = insertelement <2 x double> undef, double %B1, i32 0
; CHECK: %X1.v.i1.2 = insertelement <2 x double> %X1.v.i1.1, double %B2, i32 1
; CHECK: %X1.v.i0.1 = insertelement <2 x double> undef, double %A1, i32 0
; CHECK: %X1.v.i0.2 = insertelement <2 x double> %X1.v.i0.1, double %A2, i32 1
	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
; CHECK: %X1 = fsub <2 x double> %X1.v.i0.2, %X1.v.i1.2
	%Y1 = fmul double %X1, %A1
	%Y2 = fmul double %X2, %A2
; CHECK: %Y1 = fmul <2 x double> %X1, %X1.v.i0.2
	%C1 = fcmp ogt double %X1, %A1
        %C2 = fcmp ogt double %X2, %A2
; CHECK: %C1 = fcmp ogt <2 x double> %X1, %X1.v.i0.2
; CHECK-NB: fcmp ogt double
        %Z1 = select i1 %C1, double %Y1, double %B1
        %Z2 = select i1 %C2, double %Y2, double %B2
; CHECK: %Z1 = select <2 x i1> %C1, <2 x double> %Y1, <2 x double> %X1.v.i1.2
	%R  = fmul double %Z1, %Z2
; CHECK: %Z1.v.r1 = extractelement <2 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <2 x double> %Z1, i32 1
; CHECK: %R = fmul double %Z1.v.r1, %Z1.v.r2
	ret double %R
; CHECK: ret double %R
}

