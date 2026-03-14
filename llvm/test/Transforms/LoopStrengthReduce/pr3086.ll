; RUN: opt < %s -loop-reduce
; RUN: opt < %s -passes='require<scalar-evolution>'
; PR 3086

	%struct.Cls = type { i32, i8, [2 x ptr], [2 x ptr] }
	%struct.Lit = type { i8 }

define fastcc i64 @collect_clauses() nounwind {
entry:
	br label %bb11

bb5:		; preds = %bb9
	%0 = load ptr, ptr %storemerge, align 8		; <ptr> [#uses=0]
	%indvar.next8 = add i64 %storemerge.rec, 1		; <i64> [#uses=1]
	br label %bb9

bb9:		; preds = %bb22, %bb5
	%storemerge.rec = phi i64 [ %indvar.next8, %bb5 ], [ 0, %bb22 ]		; <i64> [#uses=2]
	%storemerge = getelementptr ptr, ptr null, i64 %storemerge.rec		; <ptr> [#uses=2]
	%1 = icmp ugt ptr null, %storemerge		; <i1> [#uses=1]
	br i1 %1, label %bb5, label %bb22

bb11:		; preds = %bb22, %entry
	%2 = load ptr, ptr null, align 8		; <ptr> [#uses=0]
	br label %bb22

bb22:		; preds = %bb11, %bb9
	br i1 false, label %bb11, label %bb9
}
