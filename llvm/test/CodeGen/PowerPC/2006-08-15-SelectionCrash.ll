; RUN: llc -verify-machineinstrs < %s
; REQUIRES: default_triple

	%struct..0anon = type { i32 }
	%struct.rtx_def = type { i16, i8, i8, [1 x %struct..0anon] }

define fastcc void @immed_double_const(i32 %i0, i32 %i1) {
entry:
	%tmp1 = load i32, ptr null		; <i32> [#uses=1]
	switch i32 %tmp1, label %bb103 [
		 i32 1, label %bb
		 i32 3, label %bb
	]
bb:		; preds = %entry, %entry
	%tmp14 = icmp sgt i32 0, 31		; <i1> [#uses=1]
	br i1 %tmp14, label %cond_next77, label %cond_next17
cond_next17:		; preds = %bb
	ret void
cond_next77:		; preds = %bb
	%tmp79.not = icmp ne i32 %i1, 0		; <i1> [#uses=1]
	%tmp84 = icmp slt i32 %i0, 0		; <i1> [#uses=2]
	%bothcond1 = or i1 %tmp79.not, %tmp84		; <i1> [#uses=1]
	br i1 %bothcond1, label %bb88, label %bb99
bb88:		; preds = %cond_next77
	%bothcond2 = and i1 false, %tmp84		; <i1> [#uses=0]
	ret void
bb99:		; preds = %cond_next77
	ret void
bb103:		; preds = %entry
	ret void
}
