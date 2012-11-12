; RUN: llc < %s -mtriple=i686-pc-linux-gnu
; PR1799

	%struct.c34007g__designated___XUB = type { i32, i32, i32, i32 }
	%struct.c34007g__pkg__parent = type { i32*, %struct.c34007g__designated___XUB* }

define void @_ada_c34007g() {
entry:
	%x8 = alloca %struct.c34007g__pkg__parent, align 8		; <%struct.c34007g__pkg__parent*> [#uses=2]
	br i1 true, label %bb1271, label %bb848

bb848:		; preds = %entry
	ret void

bb1271:		; preds = %bb898
	%tmp1272 = getelementptr %struct.c34007g__pkg__parent* %x8, i32 0, i32 0		; <i32**> [#uses=1]
	%x82167 = bitcast %struct.c34007g__pkg__parent* %x8 to i64*		; <i64*> [#uses=1]
	br i1 true, label %bb4668, label %bb848

bb4668:		; preds = %bb4648
	%tmp5464 = load i64* %x82167, align 8		; <i64> [#uses=1]
	%tmp5467 = icmp ne i64 0, %tmp5464		; <i1> [#uses=1]
	%tmp5470 = load i32** %tmp1272, align 8		; <i32*> [#uses=1]
	%tmp5471 = icmp eq i32* %tmp5470, null		; <i1> [#uses=1]
	call fastcc void @c34007g__pkg__create.311( %struct.c34007g__pkg__parent* null, i32 7, i32 9, i32 2, i32 4, i32 1 )
	%tmp5475 = or i1 %tmp5471, %tmp5467		; <i1> [#uses=1]
	%tmp5497 = or i1 %tmp5475, false		; <i1> [#uses=1]
	br i1 %tmp5497, label %bb848, label %bb5507

bb5507:		; preds = %bb4668
	ret void

}

declare fastcc void @c34007g__pkg__create.311(%struct.c34007g__pkg__parent*, i32, i32, i32, i32, i32)
