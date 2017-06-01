; RUN: llc -verify-machineinstrs < %s > %t
; ModuleID = '<stdin>'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-apple-darwin8"

define ppc_fp128 @plus(ppc_fp128 %x, ppc_fp128 %y) {
entry:
	%tmp3 = fadd ppc_fp128 %x, %y		; <ppc_fp128> [#uses=1]
	ret ppc_fp128 %tmp3
}

define ppc_fp128 @minus(ppc_fp128 %x, ppc_fp128 %y) {
entry:
	%tmp3 = fsub ppc_fp128 %x, %y		; <ppc_fp128> [#uses=1]
	ret ppc_fp128 %tmp3
}

define ppc_fp128 @times(ppc_fp128 %x, ppc_fp128 %y) {
entry:
	%tmp3 = fmul ppc_fp128 %x, %y		; <ppc_fp128> [#uses=1]
	ret ppc_fp128 %tmp3
}

define ppc_fp128 @divide(ppc_fp128 %x, ppc_fp128 %y) {
entry:
	%tmp3 = fdiv ppc_fp128 %x, %y		; <ppc_fp128> [#uses=1]
	ret ppc_fp128 %tmp3
}

