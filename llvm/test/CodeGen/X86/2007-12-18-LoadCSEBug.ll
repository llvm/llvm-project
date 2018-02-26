; RUN: llc < %s -mtriple=i686-- -mcpu=generic | grep "(%esp)" | count 2
; PR1872

	%struct.c34007g__designated___XUB = type { i32, i32, i32, i32 }
	%struct.c34007g__pkg__parent = type { i32*, %struct.c34007g__designated___XUB* }

define void @_ada_c34007g() {
entry:
	%x8 = alloca %struct.c34007g__pkg__parent, align 8		; <%struct.c34007g__pkg__parent*> [#uses=2]
	%tmp1272 = getelementptr %struct.c34007g__pkg__parent, %struct.c34007g__pkg__parent* %x8, i32 0, i32 0		; <i32**> [#uses=1]
	%x82167 = bitcast %struct.c34007g__pkg__parent* %x8 to i64*		; <i64*> [#uses=1]
	br i1 true, label %bb4668, label %bb848

bb4668:		; preds = %bb4648
	%tmp5464 = load i64, i64* %x82167, align 8		; <i64> [#uses=1]
	%tmp5467 = icmp ne i64 0, %tmp5464		; <i1> [#uses=1]
	%tmp5470 = load i32*, i32** %tmp1272, align 8		; <i32*> [#uses=1]
	%tmp5471 = icmp eq i32* %tmp5470, null		; <i1> [#uses=1]
	%tmp5475 = or i1 %tmp5471, %tmp5467		; <i1> [#uses=1]
	%tmp5497 = or i1 %tmp5475, false		; <i1> [#uses=1]
	br i1 %tmp5497, label %bb848, label %bb5507

bb848:		; preds = %entry
	ret void

bb5507:		; preds = %bb4668
	ret void
}
