target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -bb-vectorize-vector-bits=192 -instcombine -gvn -S | FileCheck %s

; Basic depth-3 chain
define double @test1(double %A1, double %A2, double %A3, double %B1, double %B2, double %B3) {
; CHECK: @test1
; CHECK: %X1.v.i1.11 = insertelement <3 x double> undef, double %B1, i32 0
; CHECK: %X1.v.i1.22 = insertelement <3 x double> %X1.v.i1.11, double %B2, i32 1
; CHECK: %X1.v.i1 = insertelement <3 x double> %X1.v.i1.22, double %B3, i32 2
; CHECK: %X1.v.i0.13 = insertelement <3 x double> undef, double %A1, i32 0
; CHECK: %X1.v.i0.24 = insertelement <3 x double> %X1.v.i0.13, double %A2, i32 1
; CHECK: %X1.v.i0 = insertelement <3 x double> %X1.v.i0.24, double %A3, i32 2
	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
	%X3 = fsub double %A3, %B3
; CHECK: %X1 = fsub <3 x double> %X1.v.i0, %X1.v.i1
	%Y1 = fmul double %X1, %A1
	%Y2 = fmul double %X2, %A2
	%Y3 = fmul double %X3, %A3
; CHECK: %Y1 = fmul <3 x double> %X1, %X1.v.i0
	%Z1 = fadd double %Y1, %B1
	%Z2 = fadd double %Y2, %B2
	%Z3 = fadd double %Y3, %B3
; CHECK: %Z1 = fadd <3 x double> %Y1, %X1.v.i1
        %R1 = fmul double %Z1, %Z2
	%R  = fmul double %R1, %Z3
; CHECK: %Z1.v.r210 = extractelement <3 x double> %Z1, i32 2
; CHECK: %Z1.v.r1 = extractelement <3 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <3 x double> %Z1, i32 1
; CHECK: %R1 = fmul double %Z1.v.r1, %Z1.v.r2
; CHECK: %R = fmul double %R1, %Z1.v.r210
	ret double %R
; CHECK: ret double %R
}

