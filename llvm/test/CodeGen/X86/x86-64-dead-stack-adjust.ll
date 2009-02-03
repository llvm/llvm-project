; RUN: llvm-as < %s | llc | not grep rsp
; RUN: llvm-as < %s | llc | grep cvttsd2siq

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"

define double @a(double %b) nounwind  {
entry:
	%tmp12 = fptoui double %b to i32		; <i32> [#uses=1]
	%tmp123 = uitofp i32 %tmp12 to double		; <double> [#uses=1]
	ret double %tmp123
}
