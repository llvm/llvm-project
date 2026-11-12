target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
; RUN: opt < %s -passes=instcombine -S | not grep "ret i1 0"
; PR1850

define i1 @test() {
	%cond = icmp ule ptr inttoptr (i64 4294967297 to ptr), inttoptr (i64 5 to ptr)
	ret i1 %cond
}
