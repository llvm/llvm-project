; RUN: llc < %s -mtriple=i686--
; PR2596

@data = external global [400 x i64]		; <ptr> [#uses=5]

define void @foo(ptr noalias, ptr noalias) {
	load i64, ptr getelementptr ([400 x i64], ptr @data, i32 0, i64 200), align 4		; <i64>:3 [#uses=1]
	load i64, ptr getelementptr ([400 x i64], ptr @data, i32 0, i64 199), align 4		; <i64>:4 [#uses=1]
	load i64, ptr getelementptr ([400 x i64], ptr @data, i32 0, i64 198), align 4		; <i64>:5 [#uses=2]
	load i64, ptr getelementptr ([400 x i64], ptr @data, i32 0, i64 197), align 4		; <i64>:6 [#uses=1]
	br i1 false, label %28, label %7

; <label>:7		; preds = %2
	load ptr, ptr getelementptr (ptr, ptr @data, i64 180), align 8		; <ptr>:8 [#uses=1]
	bitcast ptr %8 to ptr		; <ptr>:9 [#uses=1]
	ptrtoint ptr %9 to i64		; <i64>:10 [#uses=1]
	mul i64 %4, %3		; <i64>:11 [#uses=1]
	add i64 0, %11		; <i64>:12 [#uses=1]
	shl i64 %12, 3		; <i64>:13 [#uses=1]
	sub i64 %10, %13		; <i64>:14 [#uses=1]
	add i64 %5, 0		; <i64>:15 [#uses=1]
	shl i64 %15, 3		; <i64>:16 [#uses=1]
	bitcast i64 %16 to i64		; <i64>:17 [#uses=1]
	mul i64 %6, %5		; <i64>:18 [#uses=1]
	add i64 0, %18		; <i64>:19 [#uses=1]
	shl i64 %19, 3		; <i64>:20 [#uses=1]
	sub i64 %17, %20		; <i64>:21 [#uses=1]
	add i64 0, %21		; <i64>:22 [#uses=1]
	add i64 0, %14		; <i64>:23 [#uses=1]
	br label %24

; <label>:24		; preds = %24, %7
	phi i64 [ 0, %24 ], [ %22, %7 ]		; <i64>:25 [#uses=1]
	phi i64 [ 0, %24 ], [ %23, %7 ]		; <i64>:26 [#uses=0]
	add i64 %25, 24		; <i64>:27 [#uses=0]
	br label %24

; <label>:28		; preds = %2
	unreachable
}
