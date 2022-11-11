; RUN: llc < %s -mtriple=x86_64--

	%struct.XX = type <{ i8 }>
	%struct.YY = type { i64 }
	%struct.ZZ = type opaque

define signext i8 @f(ptr %fontMap, ptr %uen)   {
entry:
	%tmp45 = add i16 0, 1		; <i16> [#uses=2]
	br i1 false, label %bb124, label %bb53

bb53:		; preds = %entry
	%tmp55 = call ptr @AA( i64 1, ptr %uen )		; <ptr> [#uses=3]
	%tmp2728128 = load ptr, ptr null		; <ptr> [#uses=1]
	%tmp61 = load ptr, ptr %tmp55, align 8		; <ptr> [#uses=1]
	%tmp62 = getelementptr %struct.YY, ptr %tmp61, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp63 = load i64, ptr %tmp62, align 8		; <i64> [#uses=1]
	%tmp6566 = zext i16 %tmp45 to i64		; <i64> [#uses=1]
	%tmp67 = shl i64 %tmp6566, 1		; <i64> [#uses=1]
	call void @BB( ptr %tmp55, i64 %tmp67, i8 signext  0, ptr %uen )
	%tmp121131 = icmp eq i16 %tmp45, 1		; <i1> [#uses=1]
	br i1 %tmp121131, label %bb124, label %bb70.preheader

bb70.preheader:		; preds = %bb53
	br label %bb70

bb70:		; preds = %bb119, %bb70.preheader
	%indvar133 = phi i32 [ %indvar.next134, %bb119 ], [ 0, %bb70.preheader ]		; <i32> [#uses=2]
	%tmp.135 = trunc i64 %tmp63 to i32		; <i32> [#uses=1]
	%tmp136 = shl i32 %indvar133, 1		; <i32> [#uses=1]
	%DD = add i32 %tmp136, %tmp.135		; <i32> [#uses=1]
	%tmp73 = load ptr, ptr %tmp2728128, align 8		; <ptr> [#uses=0]
	br i1 false, label %bb119, label %bb77

bb77:		; preds = %bb70
	%tmp8384 = trunc i32 %DD to i16		; <i16> [#uses=1]
	%tmp85 = sub i16 0, %tmp8384		; <i16> [#uses=1]
	store i16 %tmp85, ptr null, align 8
	call void @CC( ptr %tmp55, i64 0, i64 2, ptr null, ptr %uen )
	ret i8 0

bb119:		; preds = %bb70
	%indvar.next134 = add i32 %indvar133, 1		; <i32> [#uses=1]
	br label %bb70

bb124:		; preds = %bb53, %entry
	ret i8 undef
}

declare ptr @AA(i64, ptr)

declare void @BB(ptr, i64, i8 signext , ptr)

declare void @CC(ptr, i64, i64, ptr, ptr)
