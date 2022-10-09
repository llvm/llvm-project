; RUN: opt < %s -licm -simple-loop-unswitch -disable-output
	%struct.III_scalefac_t = type { [22 x i32], [13 x [3 x i32]] }
	%struct.gr_info = type { i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, [4 x i32] }

define i32 @scale_bitcount_lsf(ptr %scalefac, ptr %cod_info) {
entry:
	br i1 false, label %bb28, label %bb133.preheader

bb133.preheader:		; preds = %entry
	ret i32 0

bb28:		; preds = %entry
	br i1 false, label %bb63.outer, label %bb79

bb63.outer:		; preds = %bb73, %bb28
	br i1 false, label %bb35, label %bb73

bb35:		; preds = %cond_next60, %bb63.outer
	%window.34 = phi i32 [ %tmp62, %cond_next60 ], [ 0, %bb63.outer ]		; <i32> [#uses=1]
	%tmp44 = getelementptr [4 x i32], ptr null, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp46 = load i32, ptr %tmp44, align 4		; <i32> [#uses=0]
	br i1 false, label %cond_true50, label %cond_next60

cond_true50:		; preds = %bb35
	%tmp59 = getelementptr [4 x i32], ptr null, i32 0, i32 0		; <ptr> [#uses=1]
	store i32 0, ptr %tmp59, align 4
	br label %cond_next60

cond_next60:		; preds = %cond_true50, %bb35
	%tmp62 = add i32 %window.34, 1		; <i32> [#uses=1]
	br i1 false, label %bb35, label %bb73

bb73:		; preds = %cond_next60, %bb63.outer
	%tmp76 = icmp slt i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp76, label %bb63.outer, label %bb79

bb79:		; preds = %bb73, %bb28
	ret i32 0
}
