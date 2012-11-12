target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth 3 -S | FileCheck %s -check-prefix=CHECK-RD3
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth 2 -S | FileCheck %s -check-prefix=CHECK-RD2

define double @test1(double %A1, double %A2, double %B1, double %B2) {
	%X1 = fsub double %A1, %B1
	%X2 = fsub double %A2, %B2
	%Y1 = fmul double %X1, %A1
	%Y2 = fmul double %X2, %A2
	%R  = fmul double %Y1, %Y2
	ret double %R
; CHECK-RD3: @test1
; CHECK-RD2: @test1
; CHECK-RD3-NOT: <2 x double>
; CHECK-RD2: <2 x double>
}

