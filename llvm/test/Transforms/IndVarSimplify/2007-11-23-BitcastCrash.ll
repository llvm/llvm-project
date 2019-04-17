; RUN: opt < %s -indvars -disable-output
; PR1814
target datalayout = "e-p:32:32-f64:32:64-i64:32:64-f80:32:32"

define void @FuncAt1938470480(i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i1, i1, i1, i1, i1, i1) {
EntryBlock:
	br label %asmBlockAt738ab7f3

asmBlockAt738ab9b0:		; preds = %asmBlockAt738ab7f3
	%.lcssa6 = phi i64 [ %23, %asmBlockAt738ab7f3 ]		; <i64> [#uses=0]
	ret void

asmBlockAt738ab7f3:		; preds = %asmBlockAt738ab7f3, %EntryBlock
	%ebp95 = phi i32 [ 128, %EntryBlock ], [ %24, %asmBlockAt738ab7f3 ]		; <i32> [#uses=2]
	sub <4 x i16> zeroinitializer, zeroinitializer		; <<4 x i16>>:22 [#uses=1]
	bitcast <4 x i16> %22 to i64		; <i64>:23 [#uses=1]
	add i32 %ebp95, -64		; <i32>:24 [#uses=1]
	icmp ult i32 %ebp95, 64		; <i1>:25 [#uses=1]
	br i1 %25, label %asmBlockAt738ab9b0, label %asmBlockAt738ab7f3
}
