; RUN: llvm-as < %s | opt -gvnpre | llvm-dis

define i64 @foo({ i32, i32 }** %__v) {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%__x.066.0 = phi { i32, i32 }* [ null, %entry ], [ null, %bb ]
	%tmp2.i.i63 = getelementptr { i32, i32 }* %__x.066.0, i32 0, i32 1
	br i1 false, label %bb, label %cond_true

cond_true:		; preds = %bb
	ret i64 0
}
