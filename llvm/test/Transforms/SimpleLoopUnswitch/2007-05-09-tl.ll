; RUN: opt < %s -passes="loop-mssa(simple-loop-unswitch)" -verify-memoryssa -disable-output
; PR1333

define void @pp_cxx_expression() {
entry:
	%tmp6 = lshr i32 0, 24		; <i32> [#uses=1]
	br label %tailrecurse

tailrecurse:		; preds = %tailrecurse, %tailrecurse, %entry
	switch i32 %tmp6, label %bb96 [
		 i32 24, label %bb10
		 i32 25, label %bb10
		 i32 28, label %bb10
		 i32 29, label %bb48
		 i32 31, label %bb48
		 i32 32, label %bb48
		 i32 33, label %bb48
		 i32 34, label %bb48
		 i32 36, label %bb15
		 i32 51, label %bb89
		 i32 52, label %bb89
		 i32 54, label %bb83
		 i32 57, label %bb59
		 i32 63, label %bb80
		 i32 64, label %bb80
		 i32 68, label %bb80
		 i32 169, label %bb75
		 i32 170, label %bb19
		 i32 171, label %bb63
		 i32 172, label %bb63
		 i32 173, label %bb67
		 i32 174, label %bb67
		 i32 175, label %bb19
		 i32 176, label %bb75
		 i32 178, label %bb59
		 i32 179, label %bb89
		 i32 180, label %bb59
		 i32 182, label %bb48
		 i32 183, label %bb48
		 i32 184, label %bb48
		 i32 185, label %bb48
		 i32 186, label %bb48
		 i32 195, label %bb48
		 i32 196, label %bb59
		 i32 197, label %bb89
		 i32 198, label %bb70
		 i32 199, label %bb59
		 i32 200, label %bb59
		 i32 201, label %bb59
		 i32 202, label %bb59
		 i32 203, label %bb75
		 i32 204, label %bb59
		 i32 205, label %tailrecurse
		 i32 210, label %tailrecurse
	]

bb10:		; preds = %tailrecurse, %tailrecurse, %tailrecurse
	ret void

bb15:		; preds = %tailrecurse
	ret void

bb19:		; preds = %tailrecurse, %tailrecurse
	ret void

bb48:		; preds = %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse
	ret void

bb59:		; preds = %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse
	ret void

bb63:		; preds = %tailrecurse, %tailrecurse
	ret void

bb67:		; preds = %tailrecurse, %tailrecurse
	ret void

bb70:		; preds = %tailrecurse
	ret void

bb75:		; preds = %tailrecurse, %tailrecurse, %tailrecurse
	ret void

bb80:		; preds = %tailrecurse, %tailrecurse, %tailrecurse
	ret void

bb83:		; preds = %tailrecurse
	ret void

bb89:		; preds = %tailrecurse, %tailrecurse, %tailrecurse, %tailrecurse
	ret void

bb96:		; preds = %tailrecurse
	ret void
}
