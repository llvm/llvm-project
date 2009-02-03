; RUN: llvm-as < %s | opt -instcombine -disable-output
; END.

define void @test() {
bb38.i:
	%varspec.0.i1014 = bitcast i64 123814269237067777 to i64		; <i64> [#uses=1]
	%locspec.0.i1015 = bitcast i32 1 to i32		; <i32> [#uses=2]
	%tmp51391.i1018 = lshr i64 %varspec.0.i1014, 16		; <i64> [#uses=1]
	%tmp51392.i1019 = trunc i64 %tmp51391.i1018 to i32		; <i32> [#uses=2]
	%tmp51392.mask.i1020 = lshr i32 %tmp51392.i1019, 29		; <i32> [#uses=1]
	%tmp7.i1021 = and i32 %tmp51392.mask.i1020, 1		; <i32> [#uses=2]
	%tmp18.i1026 = lshr i32 %tmp51392.i1019, 31		; <i32> [#uses=2]
	%tmp18.i1027 = trunc i32 %tmp18.i1026 to i8		; <i8> [#uses=1]
	br i1 false, label %cond_false1148.i1653, label %bb377.i1259

bb377.i1259:		; preds = %bb38.i
	br i1 false, label %cond_true541.i1317, label %cond_false1148.i1653

cond_true541.i1317:		; preds = %bb377.i1259
	%tmp545.i1318 = lshr i32 %locspec.0.i1015, 10		; <i32> [#uses=1]
	%tmp550.i1319 = lshr i32 %locspec.0.i1015, 4		; <i32> [#uses=1]
	%tmp550551.i1320 = and i32 %tmp550.i1319, 63		; <i32> [#uses=1]
	%tmp553.i1321 = icmp ult i32 %tmp550551.i1320, 4		; <i1> [#uses=1]
	%tmp558.i1322 = icmp eq i32 %tmp7.i1021, 0		; <i1> [#uses=1]
	%bothcond.i1326 = or i1 %tmp553.i1321, false		; <i1> [#uses=1]
	%bothcond1.i1327 = or i1 %bothcond.i1326, false		; <i1> [#uses=1]
	%bothcond2.not.i1328 = or i1 %bothcond1.i1327, false		; <i1> [#uses=1]
	%bothcond3.i1329 = or i1 %bothcond2.not.i1328, %tmp558.i1322		; <i1> [#uses=0]
	br i1 false, label %cond_true583.i1333, label %cond_next592.i1337

cond_true583.i1333:		; preds = %cond_true541.i1317
	br i1 false, label %cond_true586.i1335, label %cond_next592.i1337

cond_true586.i1335:		; preds = %cond_true583.i1333
	br label %cond_true.i

cond_next592.i1337:		; preds = %cond_true583.i1333, %cond_true541.i1317
	%mask_z.0.i1339 = phi i32 [ %tmp18.i1026, %cond_true541.i1317 ], [ 0, %cond_true583.i1333 ]		; <i32> [#uses=0]
	%tmp594.i1340 = and i32 %tmp545.i1318, 15		; <i32> [#uses=0]
	br label %cond_true.i

cond_false1148.i1653:		; preds = %bb377.i1259, %bb38.i
	%tmp1150.i1654 = icmp eq i32 %tmp7.i1021, 0		; <i1> [#uses=1]
	%tmp1160.i1656 = icmp eq i8 %tmp18.i1027, 0		; <i1> [#uses=1]
	%bothcond8.i1658 = or i1 %tmp1150.i1654, %tmp1160.i1656		; <i1> [#uses=1]
	%bothcond9.i1659 = or i1 %bothcond8.i1658, false		; <i1> [#uses=0]
	br label %cond_true.i

cond_true.i:		; preds = %cond_false1148.i1653, %cond_next592.i1337, %cond_true586.i1335
	ret void
}
