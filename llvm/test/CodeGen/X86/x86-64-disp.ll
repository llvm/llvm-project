; RUN: llvm-as < %s | llc -march=x86-64 | grep mov | count 2

; Fold an offset into an address even if it's not a 32-bit
; signed integer.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@call_used_regs = external global [53 x i8], align 32

define fastcc void @foo() nounwind {
	%t = getelementptr [53 x i8]* @call_used_regs, i64 0, i64 4294967295
	store i8 1, i8* %t, align 1
	ret void
}
