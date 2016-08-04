; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin

	%struct.Handle = type { %struct.oopDesc** }
	%struct.JNI_ArgumentPusher = type { %struct.SignatureIterator, %struct.JavaCallArguments* }
	%struct.JNI_ArgumentPusherArray = type { %struct.JNI_ArgumentPusher, %struct.JvmtiEventEnabled* }
	%struct.JavaCallArguments = type { [9 x i32], [9 x i32], i32*, i32*, i32, i32, i32 }
	%struct.JvmtiEventEnabled = type { i64 }
	%struct.KlassHandle = type { %struct.Handle }
	%struct.SignatureIterator = type { i32 (...)**, %struct.KlassHandle, i32, i32, i32 }
	%struct.instanceOopDesc = type { %struct.oopDesc }
	%struct.oopDesc = type { %struct.instanceOopDesc*, %struct.instanceOopDesc* }
@.str = external constant [44 x i8]		; <[44 x i8]*> [#uses=1]

define void @_ZN23JNI_ArgumentPusherArray7iterateEy(%struct.JNI_ArgumentPusherArray* %this, i64 %fingerprint) nounwind  {
entry:
	br label %bb113

bb22.preheader:		; preds = %bb113
	ret void

bb32.preheader:		; preds = %bb113
	ret void

bb42.preheader:		; preds = %bb113
	ret void

bb52:		; preds = %bb113
	br label %bb113

bb62.preheader:		; preds = %bb113
	ret void

bb72.preheader:		; preds = %bb113
	ret void

bb82:		; preds = %bb113
	br label %bb113

bb93:		; preds = %bb113
	br label %bb113

bb103.preheader:		; preds = %bb113
	ret void

bb113:		; preds = %bb113, %bb93, %bb82, %bb52, %entry
	%fingerprint_addr.0.reg2mem.9 = phi i64 [ 0, %entry ], [ 0, %bb52 ], [ 0, %bb82 ], [ 0, %bb93 ], [ %tmp118, %bb113 ]		; <i64> [#uses=1]
	tail call void @_Z28report_should_not_reach_herePKci( i8* getelementptr ([44 x i8], [44 x i8]* @.str, i32 0, i32 0), i32 817 ) nounwind 
	%tmp118 = lshr i64 %fingerprint_addr.0.reg2mem.9, 4		; <i64> [#uses=2]
	%tmp21158 = and i64 %tmp118, 15		; <i64> [#uses=1]
	switch i64 %tmp21158, label %bb113 [
		 i64 1, label %bb22.preheader
		 i64 2, label %bb52
		 i64 3, label %bb32.preheader
		 i64 4, label %bb42.preheader
		 i64 5, label %bb62.preheader
		 i64 6, label %bb82
		 i64 7, label %bb93
		 i64 8, label %bb103.preheader
		 i64 9, label %bb72.preheader
		 i64 10, label %UnifiedReturnBlock
	]

UnifiedReturnBlock:		; preds = %bb113
	ret void
}

declare void @_Z28report_should_not_reach_herePKci(i8*, i32)
