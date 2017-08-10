; RUN: llc < %s | FileCheck %s
; CHECK: movnt
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

!0 = !{ i32 1 }

define void @sub_(i32* noalias %n) {
"file movnt.f90, line 2, bb1":
	%n1 = alloca i32*, align 8
	%i = alloca i32, align 4
	%"$LCS_0" = alloca i64, align 8
	%"$LCS_S2" = alloca <2 x double>, align 16
	%r9 = load <2 x double>, <2 x double>* %"$LCS_S2", align 8
	%r10 = load i64, i64* %"$LCS_0", align 8
	%r11 = inttoptr i64 %r10 to <2 x double>*
	store <2 x double> %r9, <2 x double>* %r11, align 16, !nontemporal !0
	br label %"file movnt.f90, line 18, bb5"

"file movnt.f90, line 18, bb5":	
	ret void
}
