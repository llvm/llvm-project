; RUN: opt < %s -passes=instcombine -S | grep icmp
; PR1646

@__gthrw_pthread_cancel = weak alias i32 (i32), ptr @pthread_cancel		; <ptr> [#uses=1]
@__gthread_active_ptr.5335 = internal constant ptr @__gthrw_pthread_cancel		; <ptr> [#uses=1]
define weak i32 @pthread_cancel(i32) {
       ret i32 0
}

define i1 @__gthread_active_p() {
entry:
	%tmp1 = load ptr, ptr @__gthread_active_ptr.5335, align 4		; <ptr> [#uses=1]
	%tmp2 = icmp ne ptr %tmp1, null		; <i1> [#uses=1]
	ret i1 %tmp2
}
