; RUN: llvm-as < %s | llc -march=x86-64 -enable-eh
; PR1632

define void @_Z1fv() {
entry:
	invoke void @_Z1gv( )
			to label %return unwind label %unwind

unwind:		; preds = %entry
	br i1 false, label %eh_then, label %cleanup20

eh_then:		; preds = %unwind
	invoke void @__cxa_end_catch( )
			to label %return unwind label %unwind10

unwind10:		; preds = %eh_then
	%eh_select13 = tail call i64 (i8*, i8*, ...)* @llvm.eh.selector.i64( i8* null, i8* bitcast (void ()* @__gxx_personality_v0 to i8*), i32 1 )		; <i32> [#uses=2]
	%tmp18 = icmp slt i64 %eh_select13, 0		; <i1> [#uses=1]
	br i1 %tmp18, label %filter, label %cleanup20

filter:		; preds = %unwind10
	unreachable

cleanup20:		; preds = %unwind10, %unwind
	%eh_selector.0 = phi i64 [ 0, %unwind ], [ %eh_select13, %unwind10 ]		; <i32> [#uses=0]
	ret void

return:		; preds = %eh_then, %entry
	ret void
}

declare void @_Z1gv()

declare i64 @llvm.eh.selector.i64(i8*, i8*, ...)

declare void @__gxx_personality_v0()

declare void @__cxa_end_catch()
