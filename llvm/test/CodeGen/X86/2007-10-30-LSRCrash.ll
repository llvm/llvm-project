; RUN: llc < %s -march=x86

define i32 @unique(i8* %full, i32 %p, i32 %len, i32 %mode, i32 %verbos, i32 %flags) {
entry:
	br i1 false, label %cond_true15, label %cond_next107

cond_true15:		; preds = %entry
	br i1 false, label %bb98.preheader, label %bb

bb:		; preds = %cond_true15
	ret i32 0

bb98.preheader:		; preds = %cond_true15
	br i1 false, label %bb103, label %bb69.outer

bb76.split:		; preds = %bb69.outer.split.split, %bb69.us208
	br i1 false, label %bb103, label %bb69.outer

bb69.outer:		; preds = %bb76.split, %bb98.preheader
	%from.0.reg2mem.0.ph.rec = phi i32 [ %tmp75.rec, %bb76.split ], [ 0, %bb98.preheader ]		; <i32> [#uses=1]
	%tmp75.rec = add i32 %from.0.reg2mem.0.ph.rec, 1		; <i32> [#uses=2]
	%tmp75 = getelementptr i8* null, i32 %tmp75.rec		; <i8*> [#uses=6]
	br i1 false, label %bb69.us208, label %bb69.outer.split.split

bb69.us208:		; preds = %bb69.outer
	switch i32 0, label %bb76.split [
		 i32 47, label %bb89
		 i32 58, label %bb89
		 i32 92, label %bb89
	]

bb69.outer.split.split:		; preds = %bb69.outer
	switch i8 0, label %bb76.split [
		 i8 47, label %bb89
		 i8 58, label %bb89
		 i8 92, label %bb89
	]

bb89:		; preds = %bb69.outer.split.split, %bb69.outer.split.split, %bb69.outer.split.split, %bb69.us208, %bb69.us208, %bb69.us208
	%tmp75.lcssa189 = phi i8* [ %tmp75, %bb69.us208 ], [ %tmp75, %bb69.us208 ], [ %tmp75, %bb69.us208 ], [ %tmp75, %bb69.outer.split.split ], [ %tmp75, %bb69.outer.split.split ], [ %tmp75, %bb69.outer.split.split ]		; <i8*> [#uses=0]
	ret i32 0

bb103:		; preds = %bb76.split, %bb98.preheader
	ret i32 0

cond_next107:		; preds = %entry
	ret i32 0
}
