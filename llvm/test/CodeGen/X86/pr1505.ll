; RUN: llvm-as < %s | llc -mcpu=i486 | not grep fldl
; PR1505

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
@G = weak global float 0.000000e+00		; <float*> [#uses=1]

define void @t1(float %F) {
entry:
	store float %F, float* @G
	ret void
}
