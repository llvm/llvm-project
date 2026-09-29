; RUN: llc < %s -mtriple=sparc

; We cannot emit jump tables on Sparc, but we should correctly handle this case.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"

define i32 @foo(i32 %f) {
entry:
	switch i32 %f, label %bb14 [
		 i32 0, label %UnifiedReturnBlock
		 i32 1, label %bb4
		 i32 2, label %bb7
		 i32 3, label %bb10
	]

bb4:		; preds = %entry
	ret i32 2

bb7:		; preds = %entry
	ret i32 5

bb10:		; preds = %entry
	ret i32 9

bb14:		; preds = %entry
	ret i32 0

UnifiedReturnBlock:		; preds = %entry
	ret i32 1
}
