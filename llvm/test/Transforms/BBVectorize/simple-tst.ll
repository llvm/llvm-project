target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -bb-vectorize-vector-bits=256 -instcombine -gvn -S | FileCheck %s

; Basic depth-3 chain (target-specific type should not vectorize)
define ppc_fp128 @test7(ppc_fp128 %A1, ppc_fp128 %A2, ppc_fp128 %B1, ppc_fp128 %B2) {
; CHECK: @test7
; CHECK-NOT: <2 x ppc_fp128>
	%X1 = fsub ppc_fp128 %A1, %B1
	%X2 = fsub ppc_fp128 %A2, %B2
	%Y1 = fmul ppc_fp128 %X1, %A1
	%Y2 = fmul ppc_fp128 %X2, %A2
	%Z1 = fadd ppc_fp128 %Y1, %B1
	%Z2 = fadd ppc_fp128 %Y2, %B2
	%R  = fmul ppc_fp128 %Z1, %Z2
	ret ppc_fp128 %R
}

