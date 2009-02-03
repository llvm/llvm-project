; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin | grep movw | not grep {, %e}

	%struct.DBC_t = type { i32, i8*, i16, %struct.DBC_t*, i8*, i8*, i8*, i8*, i8*, %struct.DBC_t*, i32, i32, i32, i32, i8*, i8*, i8*, i8*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i16, i16, i32*, i8, i16, %struct.DRVOPT*, i16 }
	%struct.DRVOPT = type { i16, i32, i8, %struct.DRVOPT* }
	%struct.GENV_t = type { i32, i8*, i16, i8*, i8*, i32, i32, i32, i32, %struct.DBC_t*, i16 }
	%struct.pthread_mutex_t = type { i32, [40 x i8] }
@iodbcdm_global_lock = external global %struct.pthread_mutex_t		; <%struct.pthread_mutex_t*> [#uses=1]

define i16 @SQLDriversW(i8* %henv, i16 zeroext  %fDir, i32* %szDrvDesc, i16 signext  %cbDrvDescMax, i16* %pcbDrvDesc, i32* %szDrvAttr, i16 signext  %cbDrvAttrMax, i16* %pcbDrvAttr) signext nounwind  {
entry:
	%tmp12 = bitcast i8* %henv to %struct.GENV_t*		; <%struct.GENV_t*> [#uses=1]
	br i1 true, label %bb28, label %bb
bb:		; preds = %entry
	ret i16 0
bb28:		; preds = %entry
	br i1 false, label %bb37, label %done
bb37:		; preds = %bb28
	%tmp46 = getelementptr %struct.GENV_t* %tmp12, i32 0, i32 10		; <i16*> [#uses=1]
	store i16 0, i16* %tmp46, align 4
	br i1 false, label %bb74, label %bb92
bb74:		; preds = %bb37
	br label %bb92
bb92:		; preds = %bb74, %bb37
	%tmp95180 = shl i16 %cbDrvAttrMax, 2		; <i16> [#uses=1]
	%tmp100178 = shl i16 %cbDrvDescMax, 2		; <i16> [#uses=1]
	%tmp113 = tail call i16 @SQLDrivers_Internal( i8* %henv, i16 zeroext  %fDir, i8* null, i16 signext  %tmp100178, i16* %pcbDrvDesc, i8* null, i16 signext  %tmp95180, i16* %pcbDrvAttr, i8 zeroext  87 ) signext nounwind 		; <i16> [#uses=1]
	br i1 false, label %done, label %bb137
bb137:		; preds = %bb92
	ret i16 0
done:		; preds = %bb92, %bb28
	%retcode.0 = phi i16 [ -2, %bb28 ], [ %tmp113, %bb92 ]		; <i16> [#uses=2]
	br i1 false, label %bb167, label %bb150
bb150:		; preds = %done
	%tmp157158 = sext i16 %retcode.0 to i32		; <i32> [#uses=1]
	tail call void @trace_SQLDriversW( i32 1, i32 %tmp157158, i8* %henv, i16 zeroext  %fDir, i32* %szDrvDesc, i16 signext  %cbDrvDescMax, i16* %pcbDrvDesc, i32* %szDrvAttr, i16 signext  %cbDrvAttrMax, i16* %pcbDrvAttr ) nounwind 
	ret i16 0
bb167:		; preds = %done
	%tmp168 = tail call i32 @pthread_mutex_unlock( %struct.pthread_mutex_t* @iodbcdm_global_lock ) nounwind 		; <i32> [#uses=0]
	ret i16 %retcode.0
}

declare i32 @pthread_mutex_unlock(%struct.pthread_mutex_t*)

declare i16 @SQLDrivers_Internal(i8*, i16 zeroext , i8*, i16 signext , i16*, i8*, i16 signext , i16*, i8 zeroext ) signext nounwind 

declare void @trace_SQLDriversW(i32, i32, i8*, i16 zeroext , i32*, i16 signext , i16*, i32*, i16 signext , i16*)
