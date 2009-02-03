; RUN: llvm-as < %s | llc | grep {movl.*%edi, %eax}
; This should be a single mov, not a load of immediate + andq.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"

define i64 @test(i64 %x) nounwind  {
entry:
	%tmp123 = and i64 %x, 4294967295		; <i64> [#uses=1]
	ret i64 %tmp123
}

