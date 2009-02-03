; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah | not grep {sub.*esp}
; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah | grep cvtsi2ss
; rdar://6034396

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define void @test(i32 %x, float* %y) nounwind  {
entry:
	lshr i32 %x, 23		; <i32>:0 [#uses=1]
	uitofp i32 %0 to float		; <float>:1 [#uses=1]
	store float %1, float* %y
	ret void
}
