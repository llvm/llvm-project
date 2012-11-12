; RUN: llc < %s -fast-isel
; radr://6772169
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10"
	%0 = type { i32, i1 }		; type %0

declare %0 @llvm.sadd.with.overflow.i32(i32, i32) nounwind

define fastcc i32 @test() nounwind {
entry:
	%tmp1 = call %0 @llvm.sadd.with.overflow.i32(i32 1, i32 0)
	%tmp2 = extractvalue %0 %tmp1, 1
	br i1 %tmp2, label %.backedge, label %BB3

BB3:
	%tmp4 = extractvalue %0 %tmp1, 0
	br label %.backedge

.backedge:
	ret i32 0
}
