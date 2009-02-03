; RUN: llvm-as < %s | llc
; XFAIL: *
; PR2356
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-apple-darwin9"

define i32 @test(i64 %x, i32* %p) nounwind {
	%asmtmp = call i32 asm "", "=r,0"(i64 0) nounwind		; <i32> [#uses=0]
	%y = add i32 %asmtmp, 1
	ret i32 %y
}
