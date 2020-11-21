; RUN: opt < %s -licm -loop-unswitch -enable-new-pm=0 -disable-output 
; PR 1589

      	%struct.QBasicAtomic = type { i32 }

define void @_ZNK5QDate9addMonthsEi(%struct.QBasicAtomic* sret(%struct.QBasicAtomic)  %agg.result, %struct.QBasicAtomic* %this, i32 %nmonths) {
entry:
	br label %cond_true90

bb16:		; preds = %cond_true90
	br i1 false, label %bb93, label %cond_true90

bb45:		; preds = %cond_true90
	br i1 false, label %bb53, label %bb58

bb53:		; preds = %bb45
	br i1 false, label %bb93, label %cond_true90

bb58:		; preds = %bb45
	store i32 0, i32* null, align 4
	br i1 false, label %cond_true90, label %bb93

cond_true90:		; preds = %bb58, %bb53, %bb16, %entry
	%nmonths_addr.016.1 = phi i32 [ %nmonths, %entry ], [ 0, %bb16 ], [ 0, %bb53 ], [ %nmonths_addr.016.1, %bb58 ]		; <i32> [#uses=2]
	%tmp14 = icmp slt i32 %nmonths_addr.016.1, -11		; <i1> [#uses=1]
	br i1 %tmp14, label %bb16, label %bb45

bb93:		; preds = %bb58, %bb53, %bb16
	ret void
}
