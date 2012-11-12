; RUN: llc < %s -mtriple=powerpc-apple-darwin

	%struct.HDescriptor = type <{ i32, i32 }>

declare void @bcopy(i8*, i8*, i32)

define i32 @main(i32 %argc, i8** %argv) {
entry:
	br i1 false, label %bb31, label %bb

bb:		; preds = %entry
	ret i32 -6

bb31:		; preds = %entry
	switch i32 0, label %bb189 [
		 i32 73, label %cond_next209
		 i32 74, label %bb74
		 i32 77, label %bb57
		 i32 78, label %cond_next209
		 i32 85, label %cond_next209
		 i32 97, label %cond_next209
		 i32 100, label %cond_next209
		 i32 107, label %cond_next209
		 i32 109, label %bb57
		 i32 112, label %bb43
		 i32 115, label %cond_next209
		 i32 117, label %bb51
	]

bb43:		; preds = %bb31
	br i1 false, label %cond_true48, label %cond_true200.critedge2117

cond_true48:		; preds = %bb43
	br i1 false, label %cond_next372, label %AllDone

bb51:		; preds = %bb31
	ret i32 0

bb57:		; preds = %bb31, %bb31
	ret i32 0

bb74:		; preds = %bb31
	ret i32 0

bb189:		; preds = %bb31
	ret i32 0

cond_true200.critedge2117:		; preds = %bb43
	ret i32 0

cond_next209:		; preds = %bb31, %bb31, %bb31, %bb31, %bb31, %bb31, %bb31
	ret i32 0

cond_next372:		; preds = %cond_true48
	switch i32 0, label %bb1728 [
		 i32 73, label %bb1723
		 i32 74, label %cond_true1700
		 i32 78, label %bb1718
		 i32 85, label %bb1713
		 i32 97, label %bb1620
		 i32 107, label %AllDone
		 i32 112, label %cond_next423
		 i32 117, label %cond_next1453
	]

cond_next423:		; preds = %cond_next372
	switch i16 0, label %cond_next691 [
		 i16 18475, label %cond_next807
		 i16 18520, label %cond_next807
	]

cond_next691:		; preds = %cond_next423
	ret i32 0

cond_next807:		; preds = %cond_next423, %cond_next423
	switch i16 0, label %cond_true1192 [
		 i16 18475, label %cond_next21.i
		 i16 18520, label %cond_next21.i
	]

cond_next21.i:		; preds = %cond_next807, %cond_next807
	br i1 false, label %cond_next934, label %free.i

free.i:		; preds = %cond_next21.i
	ret i32 0

cond_next934:		; preds = %bb1005, %cond_next21.i
	%listsize.1 = phi i32 [ 0, %bb1005 ], [ 64, %cond_next21.i ]		; <i32> [#uses=1]
	%catalogExtents.2 = phi %struct.HDescriptor* [ %catalogExtents.1.reg2mem.1, %bb1005 ], [ null, %cond_next21.i ]		; <%struct.HDescriptor*> [#uses=3]
	br i1 false, label %cond_next942, label %Return1020

cond_next942:		; preds = %cond_next934
	br i1 false, label %bb1005, label %bb947

bb947:		; preds = %cond_next971, %cond_next942
	%indvar = phi i32 [ 0, %cond_next942 ], [ %indvar.next2140, %cond_next971 ]		; <i32> [#uses=2]
	%catalogExtents.1.reg2mem.0 = phi %struct.HDescriptor* [ %catalogExtents.2, %cond_next942 ], [ %tmp977978, %cond_next971 ]		; <%struct.HDescriptor*> [#uses=1]
	%extents.0.reg2mem.0 = phi %struct.HDescriptor* [ null, %cond_next942 ], [ %tmp977978, %cond_next971 ]		; <%struct.HDescriptor*> [#uses=1]
	br i1 false, label %cond_next971, label %Return1020

cond_next971:		; preds = %bb947
	%tmp = shl i32 %indvar, 6		; <i32> [#uses=1]
	%listsize.0.reg2mem.0 = add i32 %tmp, %listsize.1		; <i32> [#uses=1]
	%tmp973 = add i32 %listsize.0.reg2mem.0, 64		; <i32> [#uses=1]
	%tmp974975 = bitcast %struct.HDescriptor* %extents.0.reg2mem.0 to i8*		; <i8*> [#uses=1]
	%tmp977 = call i8* @realloc( i8* %tmp974975, i32 %tmp973 )		; <i8*> [#uses=1]
	%tmp977978 = bitcast i8* %tmp977 to %struct.HDescriptor*		; <%struct.HDescriptor*> [#uses=3]
	call void @bcopy( i8* null, i8* null, i32 64 )
	%indvar.next2140 = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 false, label %bb1005, label %bb947

bb1005:		; preds = %cond_next971, %cond_next942
	%catalogExtents.1.reg2mem.1 = phi %struct.HDescriptor* [ %catalogExtents.2, %cond_next942 ], [ %tmp977978, %cond_next971 ]		; <%struct.HDescriptor*> [#uses=2]
	br i1 false, label %Return1020, label %cond_next934

Return1020:		; preds = %bb1005, %bb947, %cond_next934
	%catalogExtents.3 = phi %struct.HDescriptor* [ %catalogExtents.1.reg2mem.0, %bb947 ], [ %catalogExtents.2, %cond_next934 ], [ %catalogExtents.1.reg2mem.1, %bb1005 ]		; <%struct.HDescriptor*> [#uses=0]
	ret i32 0

cond_true1192:		; preds = %cond_next807
	ret i32 0

cond_next1453:		; preds = %cond_next372
	ret i32 0

bb1620:		; preds = %cond_next372
	ret i32 0

cond_true1700:		; preds = %cond_next372
	ret i32 0

bb1713:		; preds = %cond_next372
	ret i32 0

bb1718:		; preds = %cond_next372
	ret i32 0

bb1723:		; preds = %cond_next372
	ret i32 0

bb1728:		; preds = %cond_next372
	ret i32 -6

AllDone:		; preds = %cond_next372, %cond_true48
	ret i32 0
}

declare i8* @realloc(i8*, i32)
