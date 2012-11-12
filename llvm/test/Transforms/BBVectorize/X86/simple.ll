target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s

; Basic depth-3 chain
define double @test1(double %A1, double %A2, double %B1, double %B2) {
	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
	%Y1 = fmul double %X1, %A1
	%Y2 = fmul double %X2, %A2
	%Z1 = fadd double %Y1, %B1
	%Z2 = fadd double %Y2, %B2
	%R  = fmul double %Z1, %Z2
	ret double %R
; CHECK: @test1
; CHECK-NOT: fmul <2 x double>
; CHECK: ret double %R
}

; Basic chain
define double @test1a(double %A1, double %A2, double %B1, double %B2) {
	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
	%Y1 = fmul double %X1, %A1
	%Y2 = fmul double %X2, %A2
	%Z1 = fadd double %Y1, %B1
	%Z2 = fadd double %Y2, %B2
	%W1 = fadd double %Y1, %Z1
	%W2 = fadd double %Y2, %Z2
	%V1 = fadd double %W1, %Z1
	%V2 = fadd double %W2, %Z2
	%Q1 = fadd double %W1, %V1
	%Q2 = fadd double %W2, %V2
	%S1 = fadd double %W1, %Q1
	%S2 = fadd double %W2, %Q2
	%R  = fmul double %S1, %S2
	ret double %R
; CHECK: @test1a
; CHECK: %X1.v.i1.1 = insertelement <2 x double> undef, double %B1, i32 0
; CHECK: %X1.v.i1.2 = insertelement <2 x double> %X1.v.i1.1, double %B2, i32 1
; CHECK: %X1.v.i0.1 = insertelement <2 x double> undef, double %A1, i32 0
; CHECK: %X1.v.i0.2 = insertelement <2 x double> %X1.v.i0.1, double %A2, i32 1
; CHECK: %X1 = fsub <2 x double> %X1.v.i0.2, %X1.v.i1.2
; CHECK: %Y1 = fmul <2 x double> %X1, %X1.v.i0.2
; CHECK: %Z1 = fadd <2 x double> %Y1, %X1.v.i1.2
; CHECK: %W1 = fadd <2 x double> %Y1, %Z1
; CHECK: %V1 = fadd <2 x double> %W1, %Z1
; CHECK: %Q1 = fadd <2 x double> %W1, %V1
; CHECK: %S1 = fadd <2 x double> %W1, %Q1
; CHECK: %S1.v.r1 = extractelement <2 x double> %S1, i32 0
; CHECK: %S1.v.r2 = extractelement <2 x double> %S1, i32 1
; CHECK: %R = fmul double %S1.v.r1, %S1.v.r2
; CHECK: ret double %R
}

; Basic depth-3 chain (last pair permuted)
define double @test2(double %A1, double %A2, double %B1, double %B2) {
	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
	%Y1 = fmul double %X1, %A1
	%Y2 = fmul double %X2, %A2
	%Z1 = fadd double %Y2, %B1
	%Z2 = fadd double %Y1, %B2
	%R  = fmul double %Z1, %Z2
	ret double %R
; CHECK: @test2
; CHECK-NOT: fmul <2 x double>
; CHECK: ret double %R
}

; Basic depth-4 chain (internal permutation)
define double @test4(double %A1, double %A2, double %B1, double %B2) {
	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
	%Y1 = fmul double %X1, %A1
	%Y2 = fmul double %X2, %A2
	%Z1 = fadd double %Y2, %B1
	%Z2 = fadd double %Y1, %B2
	%W1 = fadd double %Y2, %Z1
	%W2 = fadd double %Y1, %Z2
	%R  = fmul double %Z1, %Z2
	ret double %R
; CHECK: @test4
; CHECK-NOT: fmul <2 x double>
; CHECK: ret double %R
}

; Basic chain with shuffles
define <8 x i8> @test6(<8 x i8> %A1, <8 x i8> %A2, <8 x i8> %B1, <8 x i8> %B2) {
	%X1 = sub <8 x i8> %A1, %B1
	%X2 = sub <8 x i8> %A2, %B2
	%Y1 = mul <8 x i8> %X1, %A1
	%Y2 = mul <8 x i8> %X2, %A2
	%Z1 = add <8 x i8> %Y1, %B1
	%Z2 = add <8 x i8> %Y2, %B2
        %Q1 = shufflevector <8 x i8> %Z1, <8 x i8> %Z2, <8 x i32> <i32 15, i32 8, i32 6, i32 1, i32 13, i32 10, i32 4, i32 3>
        %Q2 = shufflevector <8 x i8> %Z2, <8 x i8> %Z2, <8 x i32> <i32 6, i32 7, i32 0, i32 1, i32 2, i32 4, i32 4, i32 1>
	%R  = mul <8 x i8> %Q1, %Q2
	ret <8 x i8> %R
; CHECK: @test6
; CHECK-NOT: sub <16 x i8>
; CHECK: ret <8 x i8>
}

