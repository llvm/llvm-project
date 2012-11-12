; RUN: llc < %s -march=x86-64

	%struct.blktkntype = type { i32, i32 }
	%struct.fieldstruc = type { [128 x i8], %struct.blktkntype*, i32, i32 }

define fastcc i32 @Env_GetFieldStruc(i8* %FieldName, i32* %Status, %struct.fieldstruc* %FieldStruc) nounwind  {
entry:
	br label %bb137.i

bb137.i:		; preds = %bb137.i, %entry
	%FieldName_addr.0209.rec.i = phi i64 [ %tmp139.rec.i, %bb137.i ], [ 0, %entry ]		; <i64> [#uses=1]
	%tmp147213.i = phi i32 [ %tmp147.i, %bb137.i ], [ 1, %entry ]		; <i32> [#uses=2]
	%tmp139.rec.i = add i64 %FieldName_addr.0209.rec.i, 1		; <i64> [#uses=2]
	%tmp141142.i = sext i32 %tmp147213.i to i64		; <i64> [#uses=0]
	%tmp147.i = add i32 %tmp147213.i, 1		; <i32> [#uses=1]
	br i1 false, label %bb137.i, label %bb149.i.loopexit

bb149.i.loopexit:		; preds = %bb137.i
	%tmp139.i = getelementptr i8* %FieldName, i64 %tmp139.rec.i		; <i8*> [#uses=0]
	unreachable
}
