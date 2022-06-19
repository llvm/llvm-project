; RUN: opt < %s -passes=instcombine -S | grep icmp
; PR1646

@__gthrw_pthread_cancel = weak alias i32 (i32), i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=1]
@__gthread_active_ptr.5335 = internal constant i8* bitcast (i32 (i32)* @__gthrw_pthread_cancel to i8*)		; <i8**> [#uses=1]
define weak i32 @pthread_cancel(i32) {
       ret i32 0
}

define i1 @__gthread_active_p() {
entry:
	%tmp1 = load i8*, i8** @__gthread_active_ptr.5335, align 4		; <i8*> [#uses=1]
	%tmp2 = icmp ne i8* %tmp1, null		; <i1> [#uses=1]
	ret i1 %tmp2
}
