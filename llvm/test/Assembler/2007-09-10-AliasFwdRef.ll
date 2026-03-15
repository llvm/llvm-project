; RUN: llvm-as  < %s | llvm-dis 
; RUN: verify-uselistorder  %s
; PR1645

@__gthread_active_ptr.5335 = internal constant ptr @__gthrw_pthread_cancel    
@__gthrw_pthread_cancel = weak alias i32 (i32), ptr @pthread_cancel



define weak i32 @pthread_cancel(i32) {
  ret i32 0
}
