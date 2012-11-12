; RUN: llc < %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc64-apple-darwin9.2.0"
	%struct.re_pattern_buffer = type <{ i8*, i64, i8, [7 x i8] }>

define i32 @xre_search_2(%struct.re_pattern_buffer* %bufp, i32 %range) nounwind  {
entry:
	br i1 false, label %bb16, label %bb49

bb16:		; preds = %entry
	%tmp19 = load i8** null, align 1		; <i8*> [#uses=1]
	%tmp21 = load i8* %tmp19, align 1		; <i8> [#uses=1]
	switch i8 %tmp21, label %bb49 [
		 i8 0, label %bb45
		 i8 1, label %bb34
	]

bb34:		; preds = %bb16
	ret i32 0

bb45:		; preds = %bb16
	ret i32 -1

bb49:		; preds = %bb16, %entry
	ret i32 0
}
