; RUN: llc < %s -march=x86-64 | grep mov | count 3

	%struct.COMPOSITE = type { i8, i16, i16 }
	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.FILE_POS = type { i8, i8, i16, i32 }
	%struct.FIRST_UNION = type { %struct.FILE_POS }
	%struct.FONT_INFO = type { %struct.metrics*, i8*, i16*, %struct.COMPOSITE*, i32, %struct.rec*, %struct.rec*, i16, i16, i16*, i8*, i8*, i16* }
	%struct.FOURTH_UNION = type { %struct.STYLE }
	%struct.GAP = type { i8, i8, i16 }
	%struct.LIST = type { %struct.rec*, %struct.rec* }
	%struct.SECOND_UNION = type { { i16, i8, i8 } }
	%struct.STYLE = type { { %struct.GAP }, { %struct.GAP }, i16, i16, i32 }
	%struct.THIRD_UNION = type { %struct.FILE*, [8 x i8] }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
	%struct.head_type = type { [2 x %struct.LIST], %struct.FIRST_UNION, %struct.SECOND_UNION, %struct.THIRD_UNION, %struct.FOURTH_UNION, %struct.rec*, { %struct.rec* }, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, i32 }
	%struct.metrics = type { i16, i16, i16, i16, i16 }
	%struct.rec = type { %struct.head_type }

define void @FontChange(i1 %foo) nounwind {
entry:
	br i1 %foo, label %bb298, label %bb49
bb49:		; preds = %entry
	ret void
bb298:		; preds = %entry
	br i1 %foo, label %bb304, label %bb366
bb304:		; preds = %bb298
	br i1 %foo, label %bb330, label %bb428
bb330:		; preds = %bb366, %bb304
	br label %bb366
bb366:		; preds = %bb330, %bb298
	br i1 %foo, label %bb330, label %bb428
bb428:		; preds = %bb366, %bb304
	br i1 %foo, label %bb650, label %bb433
bb433:		; preds = %bb428
	ret void
bb650:		; preds = %bb650, %bb428
	%tmp658 = load i8, i8* null, align 8		; <i8> [#uses=1]
	%tmp659 = icmp eq i8 %tmp658, 0		; <i1> [#uses=1]
	br i1 %tmp659, label %bb650, label %bb662
bb662:		; preds = %bb650
	br label %bb761
bb688:		; preds = %bb662
	ret void
bb761:		; preds = %bb662
	%tmp487248736542 = load i32, i32* null, align 4		; <i32> [#uses=2]
	%tmp487648776541 = and i32 %tmp487248736542, 57344		; <i32> [#uses=1]
	%tmp4881 = icmp eq i32 %tmp487648776541, 8192		; <i1> [#uses=1]
	br i1 %tmp4881, label %bb4884, label %bb4897
bb4884:		; preds = %bb761
	%tmp488948906540 = and i32 %tmp487248736542, 7168		; <i32> [#uses=1]
	%tmp4894 = icmp eq i32 %tmp488948906540, 1024		; <i1> [#uses=1]
	br i1 %tmp4894, label %bb4932, label %bb4897
bb4897:		; preds = %bb4884, %bb761
	ret void
bb4932:		; preds = %bb4884
	%tmp4933 = load i32, i32* null, align 4		; <i32> [#uses=1]
	br i1 %foo, label %bb5054, label %bb4940
bb4940:		; preds = %bb4932
	%tmp4943 = load i32, i32* null, align 4		; <i32> [#uses=2]
	switch i32 %tmp4933, label %bb5054 [
		 i32 159, label %bb4970
		 i32 160, label %bb5002
	]
bb4970:		; preds = %bb4940
	%tmp49746536 = trunc i32 %tmp4943 to i16		; <i16> [#uses=1]
	%tmp49764977 = and i16 %tmp49746536, 4095		; <i16> [#uses=1]
	%mask498049814982 = zext i16 %tmp49764977 to i64		; <i64> [#uses=1]
	%tmp4984 = getelementptr %struct.FONT_INFO, %struct.FONT_INFO* null, i64 %mask498049814982, i32 5		; <%struct.rec**> [#uses=1]
	%tmp4985 = load %struct.rec*, %struct.rec** %tmp4984, align 8		; <%struct.rec*> [#uses=1]
	%tmp4988 = getelementptr %struct.rec, %struct.rec* %tmp4985, i64 0, i32 0, i32 3		; <%struct.THIRD_UNION*> [#uses=1]
	%tmp4991 = bitcast %struct.THIRD_UNION* %tmp4988 to i32*		; <i32*> [#uses=1]
	%tmp4992 = load i32, i32* %tmp4991, align 8		; <i32> [#uses=1]
	%tmp49924993 = trunc i32 %tmp4992 to i16		; <i16> [#uses=1]
	%tmp4996 = add i16 %tmp49924993, 0		; <i16> [#uses=1]
	br label %bb5054
bb5002:		; preds = %bb4940
	%tmp50066537 = trunc i32 %tmp4943 to i16		; <i16> [#uses=1]
	%tmp50085009 = and i16 %tmp50066537, 4095		; <i16> [#uses=1]
	%mask501250135014 = zext i16 %tmp50085009 to i64		; <i64> [#uses=1]
	%tmp5016 = getelementptr %struct.FONT_INFO, %struct.FONT_INFO* null, i64 %mask501250135014, i32 5		; <%struct.rec**> [#uses=1]
	%tmp5017 = load %struct.rec*, %struct.rec** %tmp5016, align 8		; <%struct.rec*> [#uses=1]
	%tmp5020 = getelementptr %struct.rec, %struct.rec* %tmp5017, i64 0, i32 0, i32 3		; <%struct.THIRD_UNION*> [#uses=1]
	%tmp5023 = bitcast %struct.THIRD_UNION* %tmp5020 to i32*		; <i32*> [#uses=1]
	%tmp5024 = load i32, i32* %tmp5023, align 8		; <i32> [#uses=1]
	%tmp50245025 = trunc i32 %tmp5024 to i16		; <i16> [#uses=1]
	%tmp5028 = sub i16 %tmp50245025, 0		; <i16> [#uses=1]
	br label %bb5054
bb5054:		; preds = %bb5002, %bb4970, %bb4940, %bb4932
	%flen.0.reg2mem.0 = phi i16 [ %tmp4996, %bb4970 ], [ %tmp5028, %bb5002 ], [ 0, %bb4932 ], [ undef, %bb4940 ]		; <i16> [#uses=0]
	ret void
}
