; RUN: llvm-as < %s | llvm-dis
; PR1645

@__gthread_active_ptr.5335 = internal constant i8* bitcast (i32 (i32)* @__gthrw_pthread_cancel to i8*)    
@__gthrw_pthread_cancel = alias weak i32 (i32)* @pthread_cancel   



declare extern_weak i32 @pthread_cancel(i32)
