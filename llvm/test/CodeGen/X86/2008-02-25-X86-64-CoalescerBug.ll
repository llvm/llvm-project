; RUN: llc < %s -mtriple=x86_64--

	%struct.XX = type <{ i8 }>
	%struct.YY = type { i64 }
	%struct.ZZ = type opaque

define signext i8 @f(%struct.XX*** %fontMap, %struct.XX* %uen)   {
entry:
	%tmp45 = add i16 0, 1		; <i16> [#uses=2]
	br i1 false, label %bb124, label %bb53

bb53:		; preds = %entry
	%tmp55 = call %struct.YY** @AA( i64 1, %struct.XX* %uen )		; <%struct.YY**> [#uses=3]
	%tmp2728128 = load %struct.XX*, %struct.XX** null		; <%struct.XX*> [#uses=1]
	%tmp61 = load %struct.YY*, %struct.YY** %tmp55, align 8		; <%struct.YY*> [#uses=1]
	%tmp62 = getelementptr %struct.YY, %struct.YY* %tmp61, i32 0, i32 0		; <i64*> [#uses=1]
	%tmp63 = load i64, i64* %tmp62, align 8		; <i64> [#uses=1]
	%tmp6566 = zext i16 %tmp45 to i64		; <i64> [#uses=1]
	%tmp67 = shl i64 %tmp6566, 1		; <i64> [#uses=1]
	call void @BB( %struct.YY** %tmp55, i64 %tmp67, i8 signext  0, %struct.XX* %uen )
	%tmp121131 = icmp eq i16 %tmp45, 1		; <i1> [#uses=1]
	br i1 %tmp121131, label %bb124, label %bb70.preheader

bb70.preheader:		; preds = %bb53
	%tmp72 = bitcast %struct.XX* %tmp2728128 to %struct.ZZ***		; <%struct.ZZ***> [#uses=1]
	br label %bb70

bb70:		; preds = %bb119, %bb70.preheader
	%indvar133 = phi i32 [ %indvar.next134, %bb119 ], [ 0, %bb70.preheader ]		; <i32> [#uses=2]
	%tmp.135 = trunc i64 %tmp63 to i32		; <i32> [#uses=1]
	%tmp136 = shl i32 %indvar133, 1		; <i32> [#uses=1]
	%DD = add i32 %tmp136, %tmp.135		; <i32> [#uses=1]
	%tmp73 = load %struct.ZZ**, %struct.ZZ*** %tmp72, align 8		; <%struct.ZZ**> [#uses=0]
	br i1 false, label %bb119, label %bb77

bb77:		; preds = %bb70
	%tmp8384 = trunc i32 %DD to i16		; <i16> [#uses=1]
	%tmp85 = sub i16 0, %tmp8384		; <i16> [#uses=1]
	store i16 %tmp85, i16* null, align 8
	call void @CC( %struct.YY** %tmp55, i64 0, i64 2, i8* null, %struct.XX* %uen )
	ret i8 0

bb119:		; preds = %bb70
	%indvar.next134 = add i32 %indvar133, 1		; <i32> [#uses=1]
	br label %bb70

bb124:		; preds = %bb53, %entry
	ret i8 undef
}

declare %struct.YY** @AA(i64, %struct.XX*)

declare void @BB(%struct.YY**, i64, i8 signext , %struct.XX*)

declare void @CC(%struct.YY**, i64, i64, i8*, %struct.XX*)
