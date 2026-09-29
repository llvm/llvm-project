; RUN: llc < %s	
%struct.rtunion = type { i64 }
	%struct.rtx_def = type { i16, i8, i8, [1 x %struct.rtunion] }
@ix86_cpu = external global i32		; <ptr> [#uses=1]
@which_alternative = external global i32		; <ptr> [#uses=3]

declare fastcc i32 @recog()

define void @athlon_fp_unit_ready_cost() {
entry:
	%tmp = icmp slt i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp, label %cond_true.i, label %cond_true

cond_true:		; preds = %entry
	ret void

cond_true.i:		; preds = %entry
	%tmp8.i = tail call fastcc i32 @recog( )		; <i32> [#uses=1]
	switch i32 %tmp8.i, label %UnifiedReturnBlock [
		 i32 -1, label %bb2063
		 i32 19, label %bb2035
		 i32 20, label %bb2035
		 i32 21, label %bb2035
		 i32 23, label %bb2035
		 i32 24, label %bb2035
		 i32 27, label %bb2035
		 i32 32, label %bb2035
		 i32 33, label %bb1994
		 i32 35, label %bb2035
		 i32 36, label %bb1994
		 i32 90, label %bb1948
		 i32 94, label %bb1948
		 i32 95, label %bb1948
		 i32 101, label %bb1648
		 i32 102, label %bb1648
		 i32 103, label %bb1648
		 i32 104, label %bb1648
		 i32 133, label %bb1419
		 i32 135, label %bb1238
		 i32 136, label %bb1238
		 i32 137, label %bb1238
		 i32 138, label %bb1238
		 i32 139, label %bb1201
		 i32 140, label %bb1201
		 i32 141, label %bb1154
		 i32 142, label %bb1126
		 i32 144, label %bb1201
		 i32 145, label %bb1126
		 i32 146, label %bb1201
		 i32 147, label %bb1126
		 i32 148, label %bb1201
		 i32 149, label %bb1126
		 i32 150, label %bb1201
		 i32 151, label %bb1126
		 i32 152, label %bb1096
		 i32 153, label %bb1096
		 i32 154, label %bb1096
		 i32 157, label %bb1096
		 i32 158, label %bb1096
		 i32 159, label %bb1096
		 i32 162, label %bb1096
		 i32 163, label %bb1096
		 i32 164, label %bb1096
		 i32 167, label %bb1201
		 i32 168, label %bb1201
		 i32 170, label %bb1201
		 i32 171, label %bb1201
		 i32 173, label %bb1201
		 i32 174, label %bb1201
		 i32 176, label %bb1201
		 i32 177, label %bb1201
		 i32 179, label %bb993
		 i32 180, label %bb993
		 i32 181, label %bb993
		 i32 182, label %bb993
		 i32 183, label %bb993
		 i32 184, label %bb993
		 i32 365, label %bb1126
		 i32 366, label %bb1126
		 i32 367, label %bb1126
		 i32 368, label %bb1126
		 i32 369, label %bb1126
		 i32 370, label %bb1126
		 i32 371, label %bb1126
		 i32 372, label %bb1126
		 i32 373, label %bb1126
		 i32 384, label %bb1126
		 i32 385, label %bb1126
		 i32 386, label %bb1126
		 i32 387, label %bb1126
		 i32 388, label %bb1126
		 i32 389, label %bb1126
		 i32 390, label %bb1126
		 i32 391, label %bb1126
		 i32 392, label %bb1126
		 i32 525, label %bb919
		 i32 526, label %bb839
		 i32 528, label %bb919
		 i32 529, label %bb839
		 i32 531, label %cond_next6.i119
		 i32 532, label %cond_next6.i97
		 i32 533, label %cond_next6.i81
		 i32 534, label %bb495
		 i32 536, label %cond_next6.i81
		 i32 537, label %cond_next6.i81
		 i32 538, label %bb396
		 i32 539, label %bb288
		 i32 541, label %bb396
		 i32 542, label %bb396
		 i32 543, label %bb396
		 i32 544, label %bb396
		 i32 545, label %bb189
		 i32 546, label %cond_next6.i
		 i32 547, label %bb189
		 i32 548, label %cond_next6.i
		 i32 549, label %bb189
		 i32 550, label %cond_next6.i
		 i32 551, label %bb189
		 i32 552, label %cond_next6.i
		 i32 553, label %bb189
		 i32 554, label %cond_next6.i
		 i32 555, label %bb189
		 i32 556, label %cond_next6.i
		 i32 557, label %bb189
		 i32 558, label %cond_next6.i
		 i32 618, label %bb40
		 i32 619, label %bb18
		 i32 620, label %bb40
		 i32 621, label %bb10
		 i32 622, label %bb10
	]

bb10:		; preds = %cond_true.i, %cond_true.i
	ret void

bb18:		; preds = %cond_true.i
	ret void

bb40:		; preds = %cond_true.i, %cond_true.i
	ret void

cond_next6.i:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb189:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb288:		; preds = %cond_true.i
	ret void

bb396:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb495:		; preds = %cond_true.i
	ret void

cond_next6.i81:		; preds = %cond_true.i, %cond_true.i, %cond_true.i
	ret void

cond_next6.i97:		; preds = %cond_true.i
	ret void

cond_next6.i119:		; preds = %cond_true.i
	%tmp.i126 = icmp eq i16 0, 78		; <i1> [#uses=1]
	br i1 %tmp.i126, label %cond_next778, label %bb802

cond_next778:		; preds = %cond_next6.i119
	%tmp781 = icmp eq i32 0, 1		; <i1> [#uses=1]
	br i1 %tmp781, label %cond_next784, label %bb790

cond_next784:		; preds = %cond_next778
	%tmp785 = load i32, ptr @ix86_cpu		; <i32> [#uses=1]
	%tmp786 = icmp eq i32 %tmp785, 5		; <i1> [#uses=1]
	br i1 %tmp786, label %UnifiedReturnBlock, label %bb790

bb790:		; preds = %cond_next784, %cond_next778
	%tmp793 = icmp eq i32 0, 1		; <i1> [#uses=0]
	ret void

bb802:		; preds = %cond_next6.i119
	ret void

bb839:		; preds = %cond_true.i, %cond_true.i
	ret void

bb919:		; preds = %cond_true.i, %cond_true.i
	ret void

bb993:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1096:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1126:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1154:		; preds = %cond_true.i
	ret void

bb1201:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1238:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1419:		; preds = %cond_true.i
	ret void

bb1648:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	%tmp1650 = load i32, ptr @which_alternative		; <i32> [#uses=1]
	switch i32 %tmp1650, label %bb1701 [
		 i32 0, label %cond_next1675
		 i32 1, label %cond_next1675
		 i32 2, label %cond_next1675
	]

cond_next1675:		; preds = %bb1648, %bb1648, %bb1648
	ret void

bb1701:		; preds = %bb1648
	%tmp1702 = load i32, ptr @which_alternative		; <i32> [#uses=1]
	switch i32 %tmp1702, label %bb1808 [
		 i32 0, label %cond_next1727
		 i32 1, label %cond_next1727
		 i32 2, label %cond_next1727
	]

cond_next1727:		; preds = %bb1701, %bb1701, %bb1701
	ret void

bb1808:		; preds = %bb1701
	%bothcond696 = or i1 false, false		; <i1> [#uses=1]
	br i1 %bothcond696, label %bb1876, label %cond_next1834

cond_next1834:		; preds = %bb1808
	ret void

bb1876:		; preds = %bb1808
	%tmp1877signed = load i32, ptr @which_alternative		; <i32> [#uses=4]
	%tmp1877 = bitcast i32 %tmp1877signed to i32		; <i32> [#uses=1]
	%bothcond699 = icmp ult i32 %tmp1877, 2		; <i1> [#uses=1]
	%tmp1888 = icmp eq i32 %tmp1877signed, 2		; <i1> [#uses=1]
	%bothcond700 = or i1 %bothcond699, %tmp1888		; <i1> [#uses=1]
	%bothcond700.not = xor i1 %bothcond700, true		; <i1> [#uses=1]
	%tmp1894 = icmp eq i32 %tmp1877signed, 3		; <i1> [#uses=1]
	%bothcond701 = or i1 %tmp1894, %bothcond700.not		; <i1> [#uses=1]
	%bothcond702 = or i1 %bothcond701, false		; <i1> [#uses=1]
	br i1 %bothcond702, label %UnifiedReturnBlock, label %cond_next1902

cond_next1902:		; preds = %bb1876
	switch i32 %tmp1877signed, label %cond_next1937 [
		 i32 0, label %bb1918
		 i32 1, label %bb1918
		 i32 2, label %bb1918
	]

bb1918:		; preds = %cond_next1902, %cond_next1902, %cond_next1902
	ret void

cond_next1937:		; preds = %cond_next1902
	ret void

bb1948:		; preds = %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1994:		; preds = %cond_true.i, %cond_true.i
	ret void

bb2035:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb2063:		; preds = %cond_true.i
	ret void

UnifiedReturnBlock:		; preds = %bb1876, %cond_next784, %cond_true.i
	%UnifiedRetVal = phi i32 [ 100, %bb1876 ], [ 100, %cond_true.i ], [ 4, %cond_next784 ]		; <i32> [#uses=0]
	ret void
}
