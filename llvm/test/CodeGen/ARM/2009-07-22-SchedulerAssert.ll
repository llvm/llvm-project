; RUN: llc -mtriple=arm-eabi %s -o /dev/null

	%struct.cli_ac_alt = type { i8, ptr, i16, i16, ptr }
	%struct.cli_ac_node = type { i8, i8, ptr, ptr, ptr }
	%struct.cli_ac_patt = type { ptr, ptr, i16, i16, i8, i32, i32, ptr, ptr, i32, i16, i16, i16, i16, ptr, i8, i16, ptr, ptr }
	%struct.cli_bm_patt = type { ptr, ptr, i16, i16, ptr, ptr, i8, ptr, i16 }
	%struct.cli_matcher = type { i16, i8, ptr, ptr, ptr, i32, i8, i8, ptr, ptr, ptr, i32, i32, i32 }

define i32 @cli_ac_addsig(ptr nocapture %root, ptr %virname, ptr %hexsig, i32 %sigid, i16 zeroext %parts, i16 zeroext %partno, i16 zeroext %type, i32 %mindist, i32 %maxdist, ptr %offset, i8 zeroext %target) nounwind {
entry:
	br i1 undef, label %bb126, label %bb1

bb1:		; preds = %entry
	br i1 undef, label %cli_calloc.exit.thread, label %cli_calloc.exit

cli_calloc.exit.thread:		; preds = %bb1
	ret i32 -114

cli_calloc.exit:		; preds = %bb1
	br i1 undef, label %bb52, label %bb4

bb4:		; preds = %cli_calloc.exit
	br i1 undef, label %bb.i, label %bb1.i3

bb.i:		; preds = %bb4
	unreachable

bb1.i3:		; preds = %bb4
	br i1 undef, label %bb2.i4, label %cli_strdup.exit

bb2.i4:		; preds = %bb1.i3
	ret i32 -114

cli_strdup.exit:		; preds = %bb1.i3
	br i1 undef, label %cli_calloc.exit54.thread, label %cli_calloc.exit54

cli_calloc.exit54.thread:		; preds = %cli_strdup.exit
	ret i32 -114

cli_calloc.exit54:		; preds = %cli_strdup.exit
	br label %bb45

cli_calloc.exit70.thread:		; preds = %bb45
	unreachable

cli_calloc.exit70:		; preds = %bb45
	br i1 undef, label %bb.i83, label %bb1.i84

bb.i83:		; preds = %cli_calloc.exit70
	unreachable

bb1.i84:		; preds = %cli_calloc.exit70
	br i1 undef, label %bb2.i85, label %bb17

bb2.i85:		; preds = %bb1.i84
	unreachable

bb17:		; preds = %bb1.i84
	br i1 undef, label %bb22, label %bb.nph

bb.nph:		; preds = %bb17
	br label %bb18

bb18:		; preds = %bb18, %bb.nph
	br i1 undef, label %bb18, label %bb22

bb22:		; preds = %bb18, %bb17
	%0 = getelementptr i8, ptr null, i32 10		; <ptr> [#uses=1]
	%1 = load i16, ptr %0, align 2		; <i16> [#uses=1]
	%2 = add i16 %1, 1		; <i16> [#uses=1]
	%3 = zext i16 %2 to i32		; <i32> [#uses=1]
	%4 = mul i32 %3, 3		; <i32> [#uses=1]
	%5 = add i32 %4, -1		; <i32> [#uses=1]
	%6 = icmp eq i32 %5, undef		; <i1> [#uses=1]
	br i1 %6, label %bb25, label %bb43.preheader

bb43.preheader:		; preds = %bb22
	br i1 undef, label %bb28, label %bb45

bb25:		; preds = %bb22
	unreachable

bb28:		; preds = %bb43.preheader
	unreachable

bb45:		; preds = %bb43.preheader, %cli_calloc.exit54
	br i1 undef, label %cli_calloc.exit70.thread, label %cli_calloc.exit70

bb52:		; preds = %cli_calloc.exit
	unreachable

bb126:		; preds = %entry
	ret i32 -117
}
