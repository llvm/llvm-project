; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin | grep movd | count 1
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin | grep movq
; PR2677

	%struct.Bigint = type { %struct.Bigint*, i32, i32, i32, i32, [1 x i32] }

define double @_Z7qstrtodPKcPS0_Pb(i8* %s00, i8** %se, i8* %ok) nounwind {
entry:
	br i1 false, label %bb151, label %bb163

bb151:		; preds = %entry
	br label %bb163

bb163:		; preds = %bb151, %entry
	%tmp366 = load double* null, align 8		; <double> [#uses=1]
	%tmp368 = mul double %tmp366, 0.000000e+00		; <double> [#uses=1]
	%tmp368226 = bitcast double %tmp368 to i64		; <i64> [#uses=1]
	br label %bb5.i

bb5.i:		; preds = %bb5.i57.i, %bb163
	%b.0.i = phi %struct.Bigint* [ null, %bb163 ], [ %tmp9.i.i41.i, %bb5.i57.i ]		; <%struct.Bigint*> [#uses=1]
	%tmp3.i7.i728 = load i32* null, align 4		; <i32> [#uses=1]
	br label %bb.i27.i

bb.i27.i:		; preds = %bb.i27.i, %bb5.i
	%tmp23.i20.i = lshr i32 0, 16		; <i32> [#uses=1]
	br i1 false, label %bb.i27.i, label %bb5.i57.i

bb5.i57.i:		; preds = %bb.i27.i
	%tmp50.i35.i = load i32* null, align 4		; <i32> [#uses=1]
	%tmp51.i36.i = add i32 %tmp50.i35.i, 1		; <i32> [#uses=2]
	%tmp2.i.i37.i = shl i32 1, %tmp51.i36.i		; <i32> [#uses=2]
	%tmp4.i.i38.i = shl i32 %tmp2.i.i37.i, 2		; <i32> [#uses=1]
	%tmp7.i.i39.i = add i32 %tmp4.i.i38.i, 28		; <i32> [#uses=1]
	%tmp8.i.i40.i = malloc i8, i32 %tmp7.i.i39.i		; <i8*> [#uses=1]
	%tmp9.i.i41.i = bitcast i8* %tmp8.i.i40.i to %struct.Bigint*		; <%struct.Bigint*> [#uses=2]
	store i32 %tmp51.i36.i, i32* null, align 8
	store i32 %tmp2.i.i37.i, i32* null, align 4
	free %struct.Bigint* %b.0.i
	store i32 %tmp23.i20.i, i32* null, align 4
	%tmp74.i61.i = add i32 %tmp3.i7.i728, 1		; <i32> [#uses=1]
	store i32 %tmp74.i61.i, i32* null, align 4
	br i1 false, label %bb5.i, label %bb7.i

bb7.i:		; preds = %bb5.i57.i
	%tmp514 = load i32* null, align 4		; <i32> [#uses=1]
	%tmp515 = sext i32 %tmp514 to i64		; <i64> [#uses=1]
	%tmp516 = shl i64 %tmp515, 2		; <i64> [#uses=1]
	%tmp517 = add i64 %tmp516, 8		; <i64> [#uses=1]
	%tmp519 = getelementptr %struct.Bigint* %tmp9.i.i41.i, i32 0, i32 3		; <i32*> [#uses=1]
	%tmp523 = bitcast i32* %tmp519 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64( i8* null, i8* %tmp523, i64 %tmp517, i32 1 )
	%tmp524136 = bitcast i64 %tmp368226 to double		; <double> [#uses=1]
	store double %tmp524136, double* null
	unreachable
}

declare void @llvm.memcpy.i64(i8*, i8*, i64, i32) nounwind
