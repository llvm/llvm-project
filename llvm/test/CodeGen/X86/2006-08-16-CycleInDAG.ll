; RUN: llc < %s -mtriple=i686--
	%struct.expr = type { %struct.rtx_def*, i32, %struct.expr*, %struct.occr*, %struct.occr*, %struct.rtx_def* }
	%struct.hash_table = type { %struct.expr**, i32, i32, i32 }
	%struct.occr = type { %struct.occr*, %struct.rtx_def*, i8, i8 }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.u = type { [1 x i64] }

define void @test() {
	%tmp = load i32, i32* null		; <i32> [#uses=1]
	%tmp8 = call i32 @hash_rtx( )		; <i32> [#uses=1]
	%tmp11 = urem i32 %tmp8, %tmp		; <i32> [#uses=1]
	br i1 false, label %cond_next, label %return

cond_next:		; preds = %0
	%gep.upgrd.1 = zext i32 %tmp11 to i64		; <i64> [#uses=1]
	%tmp17 = getelementptr %struct.expr*, %struct.expr** null, i64 %gep.upgrd.1		; <%struct.expr**> [#uses=0]
	ret void

return:		; preds = %0
	ret void
}

declare i32 @hash_rtx()
