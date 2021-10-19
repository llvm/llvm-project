; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output
; END.

define void @main(i32 %c) {
entry:
	%tmp.9 = icmp eq i32 %c, 2		; <i1> [#uses=1]
	br i1 %tmp.9, label %endif.0, label %then.0
then.0:		; preds = %entry
	ret void
endif.0:		; preds = %entry
	br i1 false, label %then.1, label %endif.1
then.1:		; preds = %endif.0
	ret void
endif.1:		; preds = %endif.0
	br i1 false, label %then.2, label %endif.2
then.2:		; preds = %endif.1
	ret void
endif.2:		; preds = %endif.1
	br i1 false, label %then.3, label %loopentry.0
then.3:		; preds = %endif.2
	ret void
loopentry.0:		; preds = %endif.2
	br i1 false, label %no_exit.0.preheader, label %loopexit.0
no_exit.0.preheader:		; preds = %loopentry.0
	br label %no_exit.0
no_exit.0:		; preds = %endif.4, %no_exit.0.preheader
	br i1 false, label %then.4, label %endif.4
then.4:		; preds = %no_exit.0
	ret void
endif.4:		; preds = %no_exit.0
	br i1 false, label %no_exit.0, label %loopexit.0.loopexit
loopexit.0.loopexit:		; preds = %endif.4
	br label %loopexit.0
loopexit.0:		; preds = %loopexit.0.loopexit, %loopentry.0
	br i1 false, label %then.5, label %loopentry.1
then.5:		; preds = %loopexit.0
	ret void
loopentry.1:		; preds = %loopexit.0
	%tmp.143 = icmp sgt i32 0, 0		; <i1> [#uses=4]
	br i1 %tmp.143, label %no_exit.1.preheader, label %loopexit.1
no_exit.1.preheader:		; preds = %loopentry.1
	br label %no_exit.1
no_exit.1:		; preds = %endif.6, %no_exit.1.preheader
	br i1 false, label %then.6, label %shortcirc_next.3
shortcirc_next.3:		; preds = %no_exit.1
	br i1 false, label %then.6, label %shortcirc_next.4
shortcirc_next.4:		; preds = %shortcirc_next.3
	br i1 false, label %then.6, label %endif.6
then.6:		; preds = %shortcirc_next.4, %shortcirc_next.3, %no_exit.1
	ret void
endif.6:		; preds = %shortcirc_next.4
	br i1 false, label %no_exit.1, label %loopexit.1.loopexit
loopexit.1.loopexit:		; preds = %endif.6
	br label %loopexit.1
loopexit.1:		; preds = %loopexit.1.loopexit, %loopentry.1
	br i1 false, label %then.i, label %loopentry.0.i
then.i:		; preds = %loopexit.1
	ret void
loopentry.0.i:		; preds = %loopexit.1
	br i1 %tmp.143, label %no_exit.0.i.preheader, label %readvector.exit
no_exit.0.i.preheader:		; preds = %loopentry.0.i
	br label %no_exit.0.i
no_exit.0.i:		; preds = %loopexit.1.i, %no_exit.0.i.preheader
	br i1 false, label %no_exit.1.i.preheader, label %loopexit.1.i
no_exit.1.i.preheader:		; preds = %no_exit.0.i
	br label %no_exit.1.i
no_exit.1.i:		; preds = %loopexit.2.i, %no_exit.1.i.preheader
	br i1 false, label %no_exit.2.i.preheader, label %loopexit.2.i
no_exit.2.i.preheader:		; preds = %no_exit.1.i
	br label %no_exit.2.i
no_exit.2.i:		; preds = %no_exit.2.i, %no_exit.2.i.preheader
	br i1 false, label %no_exit.2.i, label %loopexit.2.i.loopexit
loopexit.2.i.loopexit:		; preds = %no_exit.2.i
	br label %loopexit.2.i
loopexit.2.i:		; preds = %loopexit.2.i.loopexit, %no_exit.1.i
	br i1 false, label %no_exit.1.i, label %loopexit.1.i.loopexit
loopexit.1.i.loopexit:		; preds = %loopexit.2.i
	br label %loopexit.1.i
loopexit.1.i:		; preds = %loopexit.1.i.loopexit, %no_exit.0.i
	br i1 false, label %no_exit.0.i, label %readvector.exit.loopexit
readvector.exit.loopexit:		; preds = %loopexit.1.i
	br label %readvector.exit
readvector.exit:		; preds = %readvector.exit.loopexit, %loopentry.0.i
	br i1 %tmp.143, label %loopentry.1.preheader.i, label %loopexit.0.i
loopentry.1.preheader.i:		; preds = %readvector.exit
	br label %loopentry.1.outer.i
loopentry.1.outer.i:		; preds = %loopexit.1.i110, %loopentry.1.preheader.i
	br label %loopentry.1.i85
loopentry.1.i85.loopexit:		; preds = %hamming.exit16.i
	br label %loopentry.1.i85
loopentry.1.i85:		; preds = %loopentry.1.i85.loopexit, %loopentry.1.outer.i
	br i1 false, label %no_exit.1.preheader.i, label %loopexit.1.i110.loopexit1
no_exit.1.preheader.i:		; preds = %loopentry.1.i85
	br label %no_exit.1.i87
no_exit.1.i87:		; preds = %then.1.i107, %no_exit.1.preheader.i
	br i1 false, label %no_exit.i.i101.preheader, label %hamming.exit.i104
no_exit.i.i101.preheader:		; preds = %no_exit.1.i87
	br label %no_exit.i.i101
no_exit.i.i101:		; preds = %no_exit.i.i101, %no_exit.i.i101.preheader
	br i1 false, label %no_exit.i.i101, label %hamming.exit.i104.loopexit
hamming.exit.i104.loopexit:		; preds = %no_exit.i.i101
	br label %hamming.exit.i104
hamming.exit.i104:		; preds = %hamming.exit.i104.loopexit, %no_exit.1.i87
	br i1 false, label %no_exit.i15.i.preheader, label %hamming.exit16.i
no_exit.i15.i.preheader:		; preds = %hamming.exit.i104
	br label %no_exit.i15.i
no_exit.i15.i:		; preds = %no_exit.i15.i, %no_exit.i15.i.preheader
	br i1 false, label %no_exit.i15.i, label %hamming.exit16.i.loopexit
hamming.exit16.i.loopexit:		; preds = %no_exit.i15.i
	br label %hamming.exit16.i
hamming.exit16.i:		; preds = %hamming.exit16.i.loopexit, %hamming.exit.i104
	br i1 false, label %loopentry.1.i85.loopexit, label %then.1.i107
then.1.i107:		; preds = %hamming.exit16.i
	br i1 false, label %no_exit.1.i87, label %loopexit.1.i110.loopexit
loopexit.1.i110.loopexit:		; preds = %then.1.i107
	br label %loopexit.1.i110
loopexit.1.i110.loopexit1:		; preds = %loopentry.1.i85
	br label %loopexit.1.i110
loopexit.1.i110:		; preds = %loopexit.1.i110.loopexit1, %loopexit.1.i110.loopexit
	br i1 false, label %loopentry.1.outer.i, label %loopexit.0.i.loopexit
loopexit.0.i.loopexit:		; preds = %loopexit.1.i110
	br label %loopexit.0.i
loopexit.0.i:		; preds = %loopexit.0.i.loopexit, %readvector.exit
	br i1 false, label %UnifiedReturnBlock.i113, label %then.2.i112
then.2.i112:		; preds = %loopexit.0.i
	br label %checkham.exit
UnifiedReturnBlock.i113:		; preds = %loopexit.0.i
	br label %checkham.exit
checkham.exit:		; preds = %UnifiedReturnBlock.i113, %then.2.i112
	br i1 false, label %loopentry.1.i14.preheader, label %loopentry.3.i.preheader
loopentry.1.i14.preheader:		; preds = %checkham.exit
	br label %loopentry.1.i14
loopentry.1.i14:		; preds = %loopexit.1.i18, %loopentry.1.i14.preheader
	br i1 false, label %no_exit.1.i16.preheader, label %loopexit.1.i18
no_exit.1.i16.preheader:		; preds = %loopentry.1.i14
	br label %no_exit.1.i16
no_exit.1.i16:		; preds = %no_exit.1.i16, %no_exit.1.i16.preheader
	br i1 false, label %no_exit.1.i16, label %loopexit.1.i18.loopexit
loopexit.1.i18.loopexit:		; preds = %no_exit.1.i16
	br label %loopexit.1.i18
loopexit.1.i18:		; preds = %loopexit.1.i18.loopexit, %loopentry.1.i14
	br i1 false, label %loopentry.1.i14, label %loopentry.3.i.loopexit
loopentry.3.i.loopexit:		; preds = %loopexit.1.i18
	br label %loopentry.3.i.preheader
loopentry.3.i.preheader:		; preds = %loopentry.3.i.loopexit, %checkham.exit
	br label %loopentry.3.i
loopentry.3.i:		; preds = %endif.1.i, %loopentry.3.i.preheader
	br i1 false, label %loopentry.4.i.preheader, label %endif.1.i
loopentry.4.i.preheader:		; preds = %loopentry.3.i
	br label %loopentry.4.i
loopentry.4.i:		; preds = %loopexit.4.i, %loopentry.4.i.preheader
	br i1 false, label %no_exit.4.i.preheader, label %loopexit.4.i
no_exit.4.i.preheader:		; preds = %loopentry.4.i
	br label %no_exit.4.i
no_exit.4.i:		; preds = %no_exit.4.i.backedge, %no_exit.4.i.preheader
	br i1 false, label %endif.0.i, label %else.i
else.i:		; preds = %no_exit.4.i
	br i1 false, label %no_exit.4.i.backedge, label %loopexit.4.i.loopexit
no_exit.4.i.backedge:		; preds = %endif.0.i, %else.i
	br label %no_exit.4.i
endif.0.i:		; preds = %no_exit.4.i
	br i1 false, label %no_exit.4.i.backedge, label %loopexit.4.i.loopexit
loopexit.4.i.loopexit:		; preds = %endif.0.i, %else.i
	br label %loopexit.4.i
loopexit.4.i:		; preds = %loopexit.4.i.loopexit, %loopentry.4.i
	br i1 false, label %loopentry.4.i, label %endif.1.i.loopexit
endif.1.i.loopexit:		; preds = %loopexit.4.i
	br label %endif.1.i
endif.1.i:		; preds = %endif.1.i.loopexit, %loopentry.3.i
	%exitcond = icmp eq i32 0, 10		; <i1> [#uses=1]
	br i1 %exitcond, label %generateT.exit, label %loopentry.3.i
generateT.exit:		; preds = %endif.1.i
	br i1 false, label %then.0.i, label %loopentry.1.i30.preheader
then.0.i:		; preds = %generateT.exit
	ret void
loopentry.1.i30.loopexit:		; preds = %loopexit.3.i
	br label %loopentry.1.i30.backedge
loopentry.1.i30.preheader:		; preds = %generateT.exit
	br label %loopentry.1.i30
loopentry.1.i30:		; preds = %loopentry.1.i30.backedge, %loopentry.1.i30.preheader
	br i1 %tmp.143, label %no_exit.0.i31.preheader, label %loopentry.1.i30.backedge
loopentry.1.i30.backedge:		; preds = %loopentry.1.i30, %loopentry.1.i30.loopexit
	br label %loopentry.1.i30
no_exit.0.i31.preheader:		; preds = %loopentry.1.i30
	br label %no_exit.0.i31
no_exit.0.i31:		; preds = %loopexit.3.i, %no_exit.0.i31.preheader
	br i1 false, label %then.1.i, label %else.0.i
then.1.i:		; preds = %no_exit.0.i31
	br i1 undef, label %then.0.i29, label %loopentry.0.i31
then.0.i29:		; preds = %then.1.i
	unreachable
loopentry.0.i31:		; preds = %then.1.i
	br i1 false, label %no_exit.0.i38.preheader, label %loopentry.1.i.preheader
no_exit.0.i38.preheader:		; preds = %loopentry.0.i31
	br label %no_exit.0.i38
no_exit.0.i38:		; preds = %no_exit.0.i38, %no_exit.0.i38.preheader
	br i1 undef, label %no_exit.0.i38, label %loopentry.1.i.preheader.loopexit
loopentry.1.i.preheader.loopexit:		; preds = %no_exit.0.i38
	br label %loopentry.1.i.preheader
loopentry.1.i.preheader:		; preds = %loopentry.1.i.preheader.loopexit, %loopentry.0.i31
	br label %loopentry.1.i
loopentry.1.i:		; preds = %endif.2.i, %loopentry.1.i.preheader
	br i1 undef, label %loopentry.2.i39.preheader, label %loopexit.1.i79.loopexit2
loopentry.2.i39.preheader:		; preds = %loopentry.1.i
	br label %loopentry.2.i39
loopentry.2.i39:		; preds = %loopexit.5.i77, %loopentry.2.i39.preheader
	br i1 false, label %loopentry.3.i40.preheader, label %hamming.exit.i71
loopentry.3.i40.preheader:		; preds = %loopentry.2.i39
	br label %loopentry.3.i40
loopentry.3.i40:		; preds = %loopexit.3.i51, %loopentry.3.i40.preheader
	br i1 false, label %no_exit.3.preheader.i42, label %loopexit.3.i51
no_exit.3.preheader.i42:		; preds = %loopentry.3.i40
	br label %no_exit.3.i49
no_exit.3.i49:		; preds = %no_exit.3.i49, %no_exit.3.preheader.i42
	br i1 undef, label %no_exit.3.i49, label %loopexit.3.i51.loopexit
loopexit.3.i51.loopexit:		; preds = %no_exit.3.i49
	br label %loopexit.3.i51
loopexit.3.i51:		; preds = %loopexit.3.i51.loopexit, %loopentry.3.i40
	br i1 undef, label %loopentry.3.i40, label %loopentry.4.i52
loopentry.4.i52:		; preds = %loopexit.3.i51
	br i1 false, label %no_exit.4.i54.preheader, label %hamming.exit.i71
no_exit.4.i54.preheader:		; preds = %loopentry.4.i52
	br label %no_exit.4.i54
no_exit.4.i54:		; preds = %no_exit.4.backedge.i, %no_exit.4.i54.preheader
	br i1 undef, label %then.1.i55, label %endif.1.i56
then.1.i55:		; preds = %no_exit.4.i54
	br i1 undef, label %no_exit.4.backedge.i, label %loopexit.4.i57
no_exit.4.backedge.i:		; preds = %endif.1.i56, %then.1.i55
	br label %no_exit.4.i54
endif.1.i56:		; preds = %no_exit.4.i54
	br i1 undef, label %no_exit.4.backedge.i, label %loopexit.4.i57
loopexit.4.i57:		; preds = %endif.1.i56, %then.1.i55
	br i1 false, label %no_exit.i.i69.preheader, label %hamming.exit.i71
no_exit.i.i69.preheader:		; preds = %loopexit.4.i57
	br label %no_exit.i.i69
no_exit.i.i69:		; preds = %no_exit.i.i69, %no_exit.i.i69.preheader
	br i1 undef, label %no_exit.i.i69, label %hamming.exit.i71.loopexit
hamming.exit.i71.loopexit:		; preds = %no_exit.i.i69
	br label %hamming.exit.i71
hamming.exit.i71:		; preds = %hamming.exit.i71.loopexit, %loopexit.4.i57, %loopentry.4.i52, %loopentry.2.i39
	br i1 undef, label %endif.2.i, label %loopentry.5.i72
loopentry.5.i72:		; preds = %hamming.exit.i71
	br i1 false, label %shortcirc_next.i74.preheader, label %loopexit.5.i77
shortcirc_next.i74.preheader:		; preds = %loopentry.5.i72
	br label %shortcirc_next.i74
shortcirc_next.i74:		; preds = %no_exit.5.i76, %shortcirc_next.i74.preheader
	br i1 undef, label %no_exit.5.i76, label %loopexit.5.i77.loopexit
no_exit.5.i76:		; preds = %shortcirc_next.i74
	br i1 undef, label %shortcirc_next.i74, label %loopexit.5.i77.loopexit
loopexit.5.i77.loopexit:		; preds = %no_exit.5.i76, %shortcirc_next.i74
	br label %loopexit.5.i77
loopexit.5.i77:		; preds = %loopexit.5.i77.loopexit, %loopentry.5.i72
	br i1 undef, label %loopentry.2.i39, label %loopexit.1.i79.loopexit
endif.2.i:		; preds = %hamming.exit.i71
	br label %loopentry.1.i
loopexit.1.i79.loopexit:		; preds = %loopexit.5.i77
	br label %loopexit.1.i79
loopexit.1.i79.loopexit2:		; preds = %loopentry.1.i
	br label %loopexit.1.i79
loopexit.1.i79:		; preds = %loopexit.1.i79.loopexit2, %loopexit.1.i79.loopexit
	br i1 undef, label %then.3.i, label %loopentry.6.i80
then.3.i:		; preds = %loopexit.1.i79
	br i1 false, label %no_exit.6.i82.preheader, label %run.exit
loopentry.6.i80:		; preds = %loopexit.1.i79
	br i1 false, label %no_exit.6.i82.preheader, label %run.exit
no_exit.6.i82.preheader:		; preds = %loopentry.6.i80, %then.3.i
	br label %no_exit.6.i82
no_exit.6.i82:		; preds = %no_exit.6.i82, %no_exit.6.i82.preheader
	br i1 undef, label %no_exit.6.i82, label %run.exit.loopexit
run.exit.loopexit:		; preds = %no_exit.6.i82
	br label %run.exit
run.exit:		; preds = %run.exit.loopexit, %loopentry.6.i80, %then.3.i
	br i1 false, label %no_exit.1.i36.preheader, label %loopentry.3.i37
else.0.i:		; preds = %no_exit.0.i31
	br i1 false, label %then.0.i4, label %loopentry.0.i6
then.0.i4:		; preds = %else.0.i
	unreachable
loopentry.0.i6:		; preds = %else.0.i
	br i1 false, label %no_exit.0.i8.preheader, label %loopentry.2.i.preheader
no_exit.0.i8.preheader:		; preds = %loopentry.0.i6
	br label %no_exit.0.i8
no_exit.0.i8:		; preds = %no_exit.0.i8, %no_exit.0.i8.preheader
	br i1 false, label %no_exit.0.i8, label %loopentry.2.i.preheader.loopexit
loopentry.2.i.preheader.loopexit:		; preds = %no_exit.0.i8
	br label %loopentry.2.i.preheader
loopentry.2.i.preheader:		; preds = %loopentry.2.i.preheader.loopexit, %loopentry.0.i6
	br label %loopentry.2.i
loopentry.2.i:		; preds = %endif.3.i19, %loopentry.2.i.preheader
	br i1 false, label %loopentry.3.i10.preheader, label %loopentry.4.i15
loopentry.3.i10.preheader:		; preds = %loopentry.2.i
	br label %loopentry.3.i10
loopentry.3.i10:		; preds = %loopexit.3.i14, %loopentry.3.i10.preheader
	br i1 false, label %no_exit.3.preheader.i, label %loopexit.3.i14
no_exit.3.preheader.i:		; preds = %loopentry.3.i10
	br label %no_exit.3.i12
no_exit.3.i12:		; preds = %no_exit.3.i12, %no_exit.3.preheader.i
	br i1 false, label %no_exit.3.i12, label %loopexit.3.i14.loopexit
loopexit.3.i14.loopexit:		; preds = %no_exit.3.i12
	br label %loopexit.3.i14
loopexit.3.i14:		; preds = %loopexit.3.i14.loopexit, %loopentry.3.i10
	br i1 false, label %loopentry.3.i10, label %loopentry.4.i15.loopexit
loopentry.4.i15.loopexit:		; preds = %loopexit.3.i14
	br label %loopentry.4.i15
loopentry.4.i15:		; preds = %loopentry.4.i15.loopexit, %loopentry.2.i
	br i1 false, label %loopentry.5.outer.i.preheader, label %loopentry.7.i
loopentry.5.outer.i.preheader:		; preds = %loopentry.4.i15
	br label %loopentry.5.outer.i
loopentry.5.outer.i:		; preds = %loopexit.5.i, %loopentry.5.outer.i.preheader
	br label %loopentry.5.i
loopentry.5.i:		; preds = %endif.1.i18, %loopentry.5.outer.i
	br i1 false, label %no_exit.5.i.preheader, label %loopexit.5.i.loopexit3
no_exit.5.i.preheader:		; preds = %loopentry.5.i
	br label %no_exit.5.i
no_exit.5.i:		; preds = %then.2.i, %no_exit.5.i.preheader
	br i1 false, label %loopentry.6.i, label %endif.1.i18
loopentry.6.i:		; preds = %no_exit.5.i
	br i1 false, label %no_exit.6.preheader.i, label %loopexit.6.i
no_exit.6.preheader.i:		; preds = %loopentry.6.i
	br label %no_exit.6.i
no_exit.6.i:		; preds = %no_exit.6.i, %no_exit.6.preheader.i
	br i1 false, label %no_exit.6.i, label %loopexit.6.i.loopexit
loopexit.6.i.loopexit:		; preds = %no_exit.6.i
	br label %loopexit.6.i
loopexit.6.i:		; preds = %loopexit.6.i.loopexit, %loopentry.6.i
	br i1 false, label %then.2.i, label %endif.1.i18
then.2.i:		; preds = %loopexit.6.i
	br i1 false, label %no_exit.5.i, label %loopexit.5.i.loopexit
endif.1.i18:		; preds = %loopexit.6.i, %no_exit.5.i
	br label %loopentry.5.i
loopexit.5.i.loopexit:		; preds = %then.2.i
	br label %loopexit.5.i
loopexit.5.i.loopexit3:		; preds = %loopentry.5.i
	br label %loopexit.5.i
loopexit.5.i:		; preds = %loopexit.5.i.loopexit3, %loopexit.5.i.loopexit
	br i1 false, label %loopentry.5.outer.i, label %loopentry.7.i.loopexit
loopentry.7.i.loopexit:		; preds = %loopexit.5.i
	br label %loopentry.7.i
loopentry.7.i:		; preds = %loopentry.7.i.loopexit, %loopentry.4.i15
	br i1 false, label %no_exit.7.i.preheader, label %hamming.exit.i
no_exit.7.i.preheader:		; preds = %loopentry.7.i
	br label %no_exit.7.i
no_exit.7.i:		; preds = %no_exit.7.i, %no_exit.7.i.preheader
	br i1 false, label %no_exit.7.i, label %loopexit.7.i
loopexit.7.i:		; preds = %no_exit.7.i
	br i1 false, label %no_exit.i.i.preheader, label %hamming.exit.i
no_exit.i.i.preheader:		; preds = %loopexit.7.i
	br label %no_exit.i.i
no_exit.i.i:		; preds = %no_exit.i.i, %no_exit.i.i.preheader
	br i1 false, label %no_exit.i.i, label %hamming.exit.i.loopexit
hamming.exit.i.loopexit:		; preds = %no_exit.i.i
	br label %hamming.exit.i
hamming.exit.i:		; preds = %hamming.exit.i.loopexit, %loopexit.7.i, %loopentry.7.i
	br i1 false, label %endif.3.i19, label %loopentry.8.i
loopentry.8.i:		; preds = %hamming.exit.i
	br i1 false, label %shortcirc_next.i.preheader, label %loopexit.8.i
shortcirc_next.i.preheader:		; preds = %loopentry.8.i
	br label %shortcirc_next.i
shortcirc_next.i:		; preds = %no_exit.8.i, %shortcirc_next.i.preheader
	br i1 false, label %no_exit.8.i, label %loopexit.8.i.loopexit
no_exit.8.i:		; preds = %shortcirc_next.i
	br i1 false, label %shortcirc_next.i, label %loopexit.8.i.loopexit
loopexit.8.i.loopexit:		; preds = %no_exit.8.i, %shortcirc_next.i
	br label %loopexit.8.i
loopexit.8.i:		; preds = %loopexit.8.i.loopexit, %loopentry.8.i
	br i1 false, label %no_exit.9.i.preheader, label %endif.3.i19
no_exit.9.i.preheader:		; preds = %loopexit.8.i
	br label %no_exit.9.i
no_exit.9.i:		; preds = %no_exit.9.i, %no_exit.9.i.preheader
	br i1 false, label %no_exit.9.i, label %endif.3.i19.loopexit
endif.3.i19.loopexit:		; preds = %no_exit.9.i
	br label %endif.3.i19
endif.3.i19:		; preds = %endif.3.i19.loopexit, %loopexit.8.i, %hamming.exit.i
	br i1 false, label %loopentry.2.i, label %loopexit.1.i20
loopexit.1.i20:		; preds = %endif.3.i19
	br i1 false, label %then.4.i, label %UnifiedReturnBlock.i
then.4.i:		; preds = %loopexit.1.i20
	br label %runcont.exit
UnifiedReturnBlock.i:		; preds = %loopexit.1.i20
	br label %runcont.exit
runcont.exit:		; preds = %UnifiedReturnBlock.i, %then.4.i
	br i1 false, label %no_exit.1.i36.preheader, label %loopentry.3.i37
no_exit.1.i36.preheader:		; preds = %runcont.exit, %run.exit
	br label %no_exit.1.i36
no_exit.1.i36:		; preds = %no_exit.1.i36, %no_exit.1.i36.preheader
	br i1 false, label %no_exit.1.i36, label %loopentry.3.i37.loopexit
loopentry.3.i37.loopexit:		; preds = %no_exit.1.i36
	br label %loopentry.3.i37
loopentry.3.i37:		; preds = %loopentry.3.i37.loopexit, %runcont.exit, %run.exit
	br i1 false, label %loopentry.4.i38.preheader, label %loopexit.3.i
loopentry.4.i38.preheader:		; preds = %loopentry.3.i37
	br label %loopentry.4.i38
loopentry.4.i38:		; preds = %loopexit.4.i42, %loopentry.4.i38.preheader
	br i1 false, label %no_exit.3.i.preheader, label %loopexit.4.i42
no_exit.3.i.preheader:		; preds = %loopentry.4.i38
	br label %no_exit.3.i
no_exit.3.i:		; preds = %no_exit.3.i.backedge, %no_exit.3.i.preheader
	br i1 false, label %endif.3.i, label %else.1.i
else.1.i:		; preds = %no_exit.3.i
	br i1 false, label %no_exit.3.i.backedge, label %loopexit.4.i42.loopexit
no_exit.3.i.backedge:		; preds = %endif.3.i, %else.1.i
	br label %no_exit.3.i
endif.3.i:		; preds = %no_exit.3.i
	br i1 false, label %no_exit.3.i.backedge, label %loopexit.4.i42.loopexit
loopexit.4.i42.loopexit:		; preds = %endif.3.i, %else.1.i
	br label %loopexit.4.i42
loopexit.4.i42:		; preds = %loopexit.4.i42.loopexit, %loopentry.4.i38
	br i1 false, label %loopentry.4.i38, label %loopexit.3.i.loopexit
loopexit.3.i.loopexit:		; preds = %loopexit.4.i42
	br label %loopexit.3.i
loopexit.3.i:		; preds = %loopexit.3.i.loopexit, %loopentry.3.i37
	%tmp.13.i155 = icmp slt i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp.13.i155, label %no_exit.0.i31, label %loopentry.1.i30.loopexit
}
