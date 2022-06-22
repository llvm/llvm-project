; RUN: llc < %s -mtriple=i686--
	%struct.expr = type { ptr, i32, ptr, ptr, ptr, ptr }
	%struct.hash_table = type { ptr, i32, i32, i32 }
	%struct.occr = type { ptr, ptr, i8, i8 }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.u = type { [1 x i64] }

define void @test() {
	%tmp = load i32, ptr null		; <i32> [#uses=1]
	%tmp8 = call i32 @hash_rtx( )		; <i32> [#uses=1]
	%tmp11 = urem i32 %tmp8, %tmp		; <i32> [#uses=1]
	br i1 false, label %cond_next, label %return

cond_next:		; preds = %0
	%gep.upgrd.1 = zext i32 %tmp11 to i64		; <i64> [#uses=1]
	%tmp17 = getelementptr ptr, ptr null, i64 %gep.upgrd.1		; <ptr> [#uses=0]
	ret void

return:		; preds = %0
	ret void
}

declare i32 @hash_rtx()
