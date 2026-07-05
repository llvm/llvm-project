; RUN: llc < %s

@letters.3100 = external constant [63 x i8]		; <ptr> [#uses=2]

define i32 @mkstemps(ptr %pattern, i32 %suffix_len, i64 %tmp42.rle) nounwind  {
bb20:
	br label %bb41

bb41:		; preds = %bb20
	%tmp8182 = trunc i64 %tmp42.rle to i32		; <i32> [#uses=1]
	%tmp83 = getelementptr [63 x i8], ptr @letters.3100, i32 0, i32 %tmp8182		; <ptr> [#uses=1]
	%tmp84 = load i8, ptr %tmp83, align 1		; <i8> [#uses=1]
	store i8 %tmp84, ptr null, align 1
	%tmp90 = urem i64 %tmp42.rle, 62		; <i64> [#uses=1]
	%tmp9091 = trunc i64 %tmp90 to i32		; <i32> [#uses=1]
	%tmp92 = getelementptr [63 x i8], ptr @letters.3100, i32 0, i32 %tmp9091		; <ptr> [#uses=1]
	store ptr %tmp92, ptr null, align 1
	ret i32 -1
}
