; RUN: opt < %s -gvn | llvm-dis

	%struct.TypHeader = type { i32, %struct.TypHeader**, [3 x i8], i8 }

define %struct.TypHeader* @LtRec(%struct.TypHeader* %hdL, %struct.TypHeader* %hdR) {
entry:
	br i1 false, label %bb556.preheader, label %bb534.preheader

bb534.preheader:		; preds = %entry
	ret %struct.TypHeader* null

bb556.preheader:		; preds = %entry
	%tmp56119 = getelementptr %struct.TypHeader* %hdR, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp56220 = load i32* %tmp56119		; <i32> [#uses=0]
	br i1 false, label %bb.nph23, label %bb675.preheader

bb.nph23:		; preds = %bb556.preheader
	ret %struct.TypHeader* null

bb656:		; preds = %bb675.outer, %bb656
	%tmp678 = load i32* %tmp677		; <i32> [#uses=0]
	br i1 false, label %bb684, label %bb656

bb684:		; preds = %bb675.outer, %bb656
	br i1 false, label %bb924.preheader, label %bb675.outer

bb675.outer:		; preds = %bb675.preheader, %bb684
	%tmp67812 = load i32* %tmp67711		; <i32> [#uses=0]
	br i1 false, label %bb684, label %bb656

bb675.preheader:		; preds = %bb556.preheader
	%tmp67711 = getelementptr %struct.TypHeader* %hdR, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp677 = getelementptr %struct.TypHeader* %hdR, i32 0, i32 0		; <i32*> [#uses=1]
	br label %bb675.outer

bb924.preheader:		; preds = %bb684
	ret %struct.TypHeader* null
}
