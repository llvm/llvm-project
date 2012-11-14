target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -bb-vectorize -S | FileCheck %s

define <4 x float> @test7(<4 x float> %A1, <4 x float> %B1, double %C1, double %C2, double %D1, double %D2) {
        %A2 = shufflevector <4 x float> %A1, <4 x float> undef, <4 x i32> <i32 2, i32 1, i32 0, i32 3>
        %B2 = shufflevector <4 x float> %B1, <4 x float> undef, <4 x i32> <i32 2, i32 1, i32 0, i32 3>
        %X1 = shufflevector <4 x float> %A2, <4 x float> undef, <2 x i32> <i32 0, i32 1>
        %X2 = shufflevector <4 x float> %B2, <4 x float> undef, <2 x i32> <i32 2, i32 3>
        %Y1 = shufflevector <2 x float> %X1, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
        %Y2 = shufflevector <2 x float> %X2, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 0, i32 1>

	%M1 = fsub double %C1, %D1
	%M2 = fsub double %C2, %D2
	%N1 = fmul double %M1, %C1
	%N2 = fmul double %M2, %C2
	%Z1 = fadd double %N1, %D1
	%Z2 = fadd double %N2, %D2

        %R = fmul <4 x float> %Y1, %Y2
        ret <4 x float> %R
; CHECK: @test7
; CHECK-NOT: <8 x float>
; CHECK: ret <4 x float>
}

