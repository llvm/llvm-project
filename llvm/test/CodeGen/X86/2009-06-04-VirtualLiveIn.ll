; RUN: llc < %s -march=x86

	%0 = type { %struct.GAP }		; type %0
	%1 = type { i16, i8, i8 }		; type %1
	%2 = type { [2 x i32], [2 x i32] }		; type %2
	%3 = type { %struct.rec* }		; type %3
	%struct.FILE_POS = type { i8, i8, i16, i32 }
	%struct.FIRST_UNION = type { %struct.FILE_POS }
	%struct.FOURTH_UNION = type { %struct.STYLE }
	%struct.GAP = type { i8, i8, i16 }
	%struct.LIST = type { %struct.rec*, %struct.rec* }
	%struct.SECOND_UNION = type { %1 }
	%struct.STYLE = type { %0, %0, i16, i16, i32 }
	%struct.THIRD_UNION = type { %2 }
	%struct.head_type = type { [2 x %struct.LIST], %struct.FIRST_UNION, %struct.SECOND_UNION, %struct.THIRD_UNION, %struct.FOURTH_UNION, %struct.rec*, %3, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, %struct.rec*, i32 }
	%struct.rec = type { %struct.head_type }

define fastcc void @MinSize(%struct.rec* %x) nounwind {
entry:
	%tmp13 = load i8* undef, align 4		; <i8> [#uses=3]
	%tmp14 = zext i8 %tmp13 to i32		; <i32> [#uses=2]
	switch i32 %tmp14, label %bb1109 [
		i32 42, label %bb246
	]

bb246:		; preds = %entry, %entry
	switch i8 %tmp13, label %bb249 [
		i8 42, label %bb269
		i8 44, label %bb269
	]

bb249:		; preds = %bb246
	%tmp3240 = icmp eq i8 %tmp13, 0		; <i1> [#uses=1]
	br i1 %tmp3240, label %bb974, label %bb269

bb269:
	%tmp3424 = getelementptr %struct.rec* %x, i32 0, i32 0, i32 0, i32 0, i32 1		; <%struct.rec**> [#uses=0]
	unreachable

bb974:
	unreachable

bb1109:		; preds = %entry
	call fastcc void @Image(i32 %tmp14) nounwind		; <i8*> [#uses=0]
	unreachable
}

declare fastcc void @Image(i32) nounwind
