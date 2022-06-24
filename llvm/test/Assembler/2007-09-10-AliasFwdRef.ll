; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0
; RUN: verify-uselistorder --opaque-pointers=0 %s
; PR1645

@__gthread_active_ptr.5335 = internal constant i8* bitcast (i32 (i32)* @__gthrw_pthread_cancel to i8*)    
@__gthrw_pthread_cancel = weak alias i32 (i32), i32 (i32)* @pthread_cancel



define weak i32 @pthread_cancel(i32) {
  ret i32 0
}
