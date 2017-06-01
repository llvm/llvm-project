; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin

define fastcc i64 @nonzero_bits1() nounwind  {
entry:
	switch i32 0, label %bb1385 [
		 i32 28, label %bb235
		 i32 35, label %bb153
		 i32 37, label %bb951
		 i32 40, label %bb289
		 i32 44, label %bb1344
		 i32 46, label %bb651
		 i32 47, label %bb651
		 i32 48, label %bb322
		 i32 49, label %bb651
		 i32 50, label %bb651
		 i32 51, label %bb651
		 i32 52, label %bb651
		 i32 53, label %bb651
		 i32 54, label %bb535
		 i32 55, label %bb565
		 i32 56, label %bb565
		 i32 58, label %bb1100
		 i32 59, label %bb1100
		 i32 60, label %bb1100
		 i32 61, label %bb1100
		 i32 63, label %bb565
		 i32 64, label %bb565
		 i32 65, label %bb565
		 i32 66, label %bb565
		 i32 73, label %bb302
		 i32 74, label %bb302
		 i32 75, label %bb302
		 i32 76, label %bb302
		 i32 77, label %bb302
		 i32 78, label %bb302
		 i32 79, label %bb302
		 i32 80, label %bb302
		 i32 81, label %bb302
		 i32 82, label %bb302
		 i32 83, label %bb302
		 i32 84, label %bb302
		 i32 85, label %bb302
		 i32 86, label %bb302
		 i32 87, label %bb302
		 i32 88, label %bb302
		 i32 89, label %bb302
		 i32 90, label %bb302
		 i32 91, label %bb507
		 i32 92, label %bb375
		 i32 93, label %bb355
		 i32 103, label %bb1277
		 i32 104, label %bb1310
		 i32 105, label %UnifiedReturnBlock
		 i32 106, label %bb1277
		 i32 107, label %bb1343
	]
bb153:		; preds = %entry
	ret i64 0
bb235:		; preds = %entry
	br i1 false, label %bb245, label %UnifiedReturnBlock
bb245:		; preds = %bb235
	ret i64 0
bb289:		; preds = %entry
	ret i64 0
bb302:		; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	ret i64 0
bb322:		; preds = %entry
	ret i64 0
bb355:		; preds = %entry
	ret i64 0
bb375:		; preds = %entry
	ret i64 0
bb507:		; preds = %entry
	ret i64 0
bb535:		; preds = %entry
	ret i64 0
bb565:		; preds = %entry, %entry, %entry, %entry, %entry, %entry
	ret i64 0
bb651:		; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry
	ret i64 0
bb951:		; preds = %entry
	ret i64 0
bb1100:		; preds = %entry, %entry, %entry, %entry
	ret i64 0
bb1277:		; preds = %entry, %entry
	br i1 false, label %UnifiedReturnBlock, label %bb1284
bb1284:		; preds = %bb1277
	ret i64 0
bb1310:		; preds = %entry
	ret i64 0
bb1343:		; preds = %entry
	ret i64 1
bb1344:		; preds = %entry
	ret i64 0
bb1385:		; preds = %entry
	ret i64 0
UnifiedReturnBlock:		; preds = %bb1277, %bb235, %entry
	%UnifiedRetVal = phi i64 [ 0, %bb235 ], [ undef, %bb1277 ], [ -1, %entry ]		; <i64> [#uses=1]
	ret i64 %UnifiedRetVal
}
