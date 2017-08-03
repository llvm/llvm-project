; RUN: llc < %s -march=x86-64

	%struct.XDesc = type <{ i32, %struct.OpaqueXDataStorageType** }>
	%struct.OpaqueXDataStorageType = type opaque

declare signext i16 @GetParamDesc(%struct.XDesc*, i32, i32, %struct.XDesc*)  

declare void @r_raise(i64, i8*, ...)

define i64 @app_send_event(i64 %self, i64 %event_class, i64 %event_id, i64 %params, i64 %need_retval) {
entry:
	br i1 false, label %cond_true109, label %bb83.preheader

bb83.preheader:		; preds = %entry
	ret i64 0

cond_true109:		; preds = %entry
	br i1 false, label %cond_next164, label %cond_true239

cond_next164:		; preds = %cond_true109
	%tmp176 = call signext i16 @GetParamDesc( %struct.XDesc* null, i32 1701999219, i32 1413830740, %struct.XDesc* null ) 
	call void (i64, i8*, ...) @r_raise( i64 0, i8* null )
	unreachable

cond_true239:		; preds = %cond_true109
	ret i64 0
}
