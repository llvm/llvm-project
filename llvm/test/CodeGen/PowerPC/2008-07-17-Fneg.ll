; RUN: llvm-as < %s | llc
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-apple-darwin9"

define hidden i64 @__fixunstfdi(ppc_fp128 %a) nounwind  {
entry:
	br i1 false, label %bb3, label %bb4

bb3:		; preds = %entry
	sub ppc_fp128 0xM80000000000000000000000000000000, 0xM00000000000000000000000000000000		; <ppc_fp128>:0 [#uses=1]
	fptoui ppc_fp128 %0 to i32		; <i32>:1 [#uses=1]
	zext i32 %1 to i64		; <i64>:2 [#uses=1]
	sub i64 0, %2		; <i64>:3 [#uses=1]
	ret i64 %3

bb4:		; preds = %entry
	ret i64 0
}
