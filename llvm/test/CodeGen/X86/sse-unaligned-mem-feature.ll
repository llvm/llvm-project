; RUN: llc -mcpu=yonah -mattr=sse-unaligned-mem -march=x86 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define <4 x float> @foo(<4 x float>* %P, <4 x float> %In) nounwind {
	%A = load <4 x float>, <4 x float>* %P, align 4
	%B = fadd <4 x float> %A, %In
	ret <4 x float> %B

; CHECK-LABEL: @foo
; CHECK:       addps (%eax), %xmm0
}
