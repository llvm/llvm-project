target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -bb-vectorize-search-limit=4 -instcombine -gvn -S | FileCheck %s -check-prefix=CHECK-SL4

define double @test1(double %A1, double %A2, double %B1, double %B2) {
; CHECK: @test1
; CHECK-SL4: @test1
; CHECK-SL4-NOT: <2 x double>
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
        ; Here we have a dependency chain: the short search limit will not
        ; see past this chain and so will not see the second part of the
        ; pair to vectorize.
        %mul41 = fmul double %Z1, %Y2
        %sub48 = fsub double %Z1, %mul41
        %mul62 = fmul double %Z1, %sub48
        %sub69 = fsub double %Z1, %mul62
        %mul83 = fmul double %Z1, %sub69
        %sub90 = fsub double %Z1, %mul83
        %mul104 = fmul double %Z1, %sub90
        %sub111 = fsub double %Z1, %mul104
        %mul125 = fmul double %Z1, %sub111
        %sub132 = fsub double %Z1, %mul125
        %mul146 = fmul double %Z1, %sub132
        %sub153 = fsub double %Z1, %mul146
        ; end of chain.
	%Z2 = fadd double %Y2, %B2
; CHECK: %Z1 = fadd <2 x double> %Y1, %X1.v.i1.2
	%R1  = fdiv double %Z1, %Z2
        %R   = fmul double %R1, %sub153
; CHECK: %Z1.v.r1 = extractelement <2 x double> %Z1, i32 0
; CHECK: %Z1.v.r2 = extractelement <2 x double> %Z1, i32 1
; CHECK: %R1 = fdiv double %Z1.v.r1, %Z1.v.r2
	ret double %R
; CHECK: ret double %R
}

