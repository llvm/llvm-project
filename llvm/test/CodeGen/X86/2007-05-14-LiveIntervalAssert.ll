; RUN: llc < %s -mtriple=x86_64--

	%struct.XDesc = type <{ i32, ptr }>
	%struct.OpaqueXDataStorageType = type opaque

declare signext i16 @GetParamDesc(ptr, i32, i32, ptr)  

declare void @r_raise(i64, ptr, ...)

define i64 @app_send_event(i64 %self, i64 %event_class, i64 %event_id, i64 %params, i64 %need_retval) {
entry:
	br i1 false, label %cond_true109, label %bb83.preheader

bb83.preheader:		; preds = %entry
	ret i64 0

cond_true109:		; preds = %entry
	br i1 false, label %cond_next164, label %cond_true239

cond_next164:		; preds = %cond_true109
	%tmp176 = call signext i16 @GetParamDesc( ptr null, i32 1701999219, i32 1413830740, ptr null ) 
	call void (i64, ptr, ...) @r_raise( i64 0, ptr null )
	unreachable

cond_true239:		; preds = %cond_true109
	ret i64 0
}
