; RUN: llc < %s -relocation-model=pic -code-model=large

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"

declare void @xscanf(i64) nounwind 

define void @foo() nounwind  {
	call void (i64)* @xscanf( i64 0 ) nounwind
	unreachable
}
