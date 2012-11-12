target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s

; Basic depth-3 chain
define double @test1(double %A1, double %A2, double %B1, double %B2) {
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
	%Z1 = fadd double %Y1, %B1
	%Z2 = fadd double %Y2, %B2
; CHECK: %Z1 = fadd <2 x double> %Y1, %X1.v.i1.2
	%R  = fmul double %Z1, %Z2
; CHECK: %Z1.v.r1 = extractelement <2 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <2 x double> %Z1, i32 1
; CHECK: %R = fmul double %Z1.v.r1, %Z1.v.r2
	ret double %R
; CHECK: ret double %R
}

; Basic depth-3 chain (last pair permuted)
define double @test2(double %A1, double %A2, double %B1, double %B2) {
; CHECK: @test2
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
	%Z1 = fadd double %Y2, %B1
	%Z2 = fadd double %Y1, %B2
; CHECK: %Z1.v.i1.1 = insertelement <2 x double> undef, double %B2, i32 0
; CHECK: %Z1.v.i1.2 = insertelement <2 x double> %Z1.v.i1.1, double %B1, i32 1
; CHECK: %Z2 = fadd <2 x double> %Y1, %Z1.v.i1.2
	%R  = fmul double %Z1, %Z2
; CHECK: %Z2.v.r1 = extractelement <2 x double> %Z2, i32 0
; CHECK: %Z2.v.r2 = extractelement <2 x double> %Z2, i32 1
; CHECK: %R = fmul double %Z2.v.r2, %Z2.v.r1
	ret double %R
; CHECK: ret double %R
}

; Basic depth-3 chain (last pair first splat)
define double @test3(double %A1, double %A2, double %B1, double %B2) {
; CHECK: @test3
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
	%Z1 = fadd double %Y2, %B1
	%Z2 = fadd double %Y2, %B2
; CHECK: %Z1.v.i0 = shufflevector <2 x double> %Y1, <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK: %Z1 = fadd <2 x double> %Z1.v.i0, %X1.v.i1.2
	%R  = fmul double %Z1, %Z2
; CHECK: %Z1.v.r1 = extractelement <2 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <2 x double> %Z1, i32 1
; CHECK: %R = fmul double %Z1.v.r1, %Z1.v.r2
	ret double %R
; CHECK: ret double %R
}

; Basic depth-3 chain (last pair second splat)
define double @test4(double %A1, double %A2, double %B1, double %B2) {
; CHECK: @test4
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
	%Z1 = fadd double %Y1, %B1
	%Z2 = fadd double %Y1, %B2
; CHECK: %Z1.v.i0 = shufflevector <2 x double> %Y1, <2 x double> undef, <2 x i32> zeroinitializer
; CHECK: %Z1 = fadd <2 x double> %Z1.v.i0, %X1.v.i1.2
	%R  = fmul double %Z1, %Z2
; CHECK: %Z1.v.r1 = extractelement <2 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <2 x double> %Z1, i32 1
; CHECK: %R = fmul double %Z1.v.r1, %Z1.v.r2
	ret double %R
; CHECK: ret double %R
}

; Basic depth-3 chain
define <2 x float> @test5(<2 x float> %A1, <2 x float> %A2, <2 x float> %B1, <2 x float> %B2) {
; CHECK: @test5
; CHECK: %X1.v.i1 = shufflevector <2 x float> %B1, <2 x float> %B2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK: %X1.v.i0 = shufflevector <2 x float> %A1, <2 x float> %A2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
	%X1 = fsub <2 x float> %A1, %B1
	%X2 = fsub <2 x float> %A2, %B2
; CHECK: %X1 = fsub <4 x float> %X1.v.i0, %X1.v.i1
	%Y1 = fmul <2 x float> %X1, %A1
	%Y2 = fmul <2 x float> %X2, %A2
; CHECK: %Y1 = fmul <4 x float> %X1, %X1.v.i0
	%Z1 = fadd <2 x float> %Y1, %B1
	%Z2 = fadd <2 x float> %Y2, %B2
; CHECK: %Z1 = fadd <4 x float> %Y1, %X1.v.i1
	%R  = fmul <2 x float> %Z1, %Z2
; CHECK: %Z1.v.r1 = shufflevector <4 x float> %Z1, <4 x float> undef, <2 x i32> <i32 0, i32 1>
; CHECK: %Z1.v.r2 = shufflevector <4 x float> %Z1, <4 x float> undef, <2 x i32> <i32 2, i32 3>
; CHECK: %R = fmul <2 x float> %Z1.v.r1, %Z1.v.r2
	ret <2 x float> %R
; CHECK: ret <2 x float> %R
}

; Basic chain with shuffles
define <8 x i8> @test6(<8 x i8> %A1, <8 x i8> %A2, <8 x i8> %B1, <8 x i8> %B2) {
; CHECK: @test6
; CHECK: %X1.v.i1 = shufflevector <8 x i8> %B1, <8 x i8> %B2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK: %X1.v.i0 = shufflevector <8 x i8> %A1, <8 x i8> %A2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
	%X1 = sub <8 x i8> %A1, %B1
	%X2 = sub <8 x i8> %A2, %B2
; CHECK: %X1 = sub <16 x i8> %X1.v.i0, %X1.v.i1
	%Y1 = mul <8 x i8> %X1, %A1
	%Y2 = mul <8 x i8> %X2, %A2
; CHECK: %Y1 = mul <16 x i8> %X1, %X1.v.i0
	%Z1 = add <8 x i8> %Y1, %B1
	%Z2 = add <8 x i8> %Y2, %B2
; CHECK: %Z1 = add <16 x i8> %Y1, %X1.v.i1
        %Q1 = shufflevector <8 x i8> %Z1, <8 x i8> %Z2, <8 x i32> <i32 15, i32 8, i32 6, i32 1, i32 13, i32 10, i32 4, i32 3>
        %Q2 = shufflevector <8 x i8> %Z2, <8 x i8> %Z2, <8 x i32> <i32 6, i32 7, i32 0, i32 1, i32 2, i32 4, i32 4, i32 1>
; CHECK: %Q1.v.i1 = shufflevector <16 x i8> %Z1, <16 x i8> undef, <16 x i32> <i32 8, i32 undef, i32 10, i32 undef, i32 undef, i32 13, i32 undef, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK: %Q1 = shufflevector <16 x i8> %Z1, <16 x i8> %Q1.v.i1, <16 x i32> <i32 23, i32 16, i32 6, i32 1, i32 21, i32 18, i32 4, i32 3, i32 14, i32 15, i32 8, i32 9, i32 10, i32 12, i32 12, i32 9>
	%R  = mul <8 x i8> %Q1, %Q2
; CHECK: %Q1.v.r1 = shufflevector <16 x i8> %Q1, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %Q1.v.r2 = shufflevector <16 x i8> %Q1, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK: %R = mul <8 x i8> %Q1.v.r1, %Q1.v.r2
	ret <8 x i8> %R
; CHECK: ret <8 x i8> %R
}

; Basic depth-3 chain (flipped order)
define double @test7(double %A1, double %A2, double %B1, double %B2) {
; CHECK: @test7
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
	%Z2 = fadd double %Y2, %B2
	%Z1 = fadd double %Y1, %B1
; CHECK: %Z1 = fadd <2 x double> %Y1, %X1.v.i1.2
	%R  = fmul double %Z1, %Z2
; CHECK: %Z1.v.r1 = extractelement <2 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <2 x double> %Z1, i32 1
; CHECK: %R = fmul double %Z1.v.r1, %Z1.v.r2
	ret double %R
; CHECK: ret double %R
}

