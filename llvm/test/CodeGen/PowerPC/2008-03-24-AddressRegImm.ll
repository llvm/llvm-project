; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--

define fastcc ptr @page_rec_get_next(ptr %rec) nounwind  {
entry:
	%tmp2627 = ptrtoint ptr %rec to i64		; <i64> [#uses=2]
	%tmp28 = and i64 %tmp2627, -16384		; <i64> [#uses=2]
	%tmp2829 = inttoptr i64 %tmp28 to ptr		; <ptr> [#uses=1]
	%tmp37 = getelementptr i8, ptr %tmp2829, i64 42		; <ptr> [#uses=1]
	%tmp40 = load i8, ptr %tmp37, align 1		; <i8> [#uses=1]
	%tmp4041 = zext i8 %tmp40 to i64		; <i64> [#uses=1]
	%tmp42 = shl i64 %tmp4041, 8		; <i64> [#uses=1]
	%tmp47 = add i64 %tmp42, 0		; <i64> [#uses=1]
	%tmp52 = and i64 %tmp47, 32768		; <i64> [#uses=1]
	%tmp72 = icmp eq i64 %tmp52, 0		; <i1> [#uses=1]
	br i1 %tmp72, label %bb91, label %bb
bb:		; preds = %entry
	ret ptr null
bb91:		; preds = %entry
	br i1 false, label %bb100, label %bb185
bb100:		; preds = %bb91
	%tmp106 = sub i64 %tmp2627, %tmp28		; <i64> [#uses=0]
	ret ptr null
bb185:		; preds = %bb91
	ret ptr null
}
