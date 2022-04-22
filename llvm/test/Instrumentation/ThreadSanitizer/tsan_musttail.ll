; To test that __tsan_func_exit always happen before musttaill call and no exception handling code.
; RUN: opt < %s -passes=tsan -S | FileCheck %s

define internal i32 @preallocated_musttail(i32* preallocated(i32) %p) sanitize_thread {
  %rv = load i32, i32* %p
  ret i32 %rv
}

define i32 @call_preallocated_musttail(i32* preallocated(i32) %a) sanitize_thread {
  %r = musttail call i32 @preallocated_musttail(i32* preallocated(i32) %a)
  ret i32 %r
}

; CHECK-LABEL:  define i32 @call_preallocated_musttail(i32* preallocated(i32) %a) 
; CHECK:          call void @__tsan_func_exit()
; CHECK-NEXT:     %r = musttail call i32 @preallocated_musttail(i32* preallocated(i32) %a)
; CHECK-NEXT:     ret i32 %r


define i32 @call_preallocated_musttail_cast(i32* preallocated(i32) %a) sanitize_thread {
  %r = musttail call i32 @preallocated_musttail(i32* preallocated(i32) %a)
  %t = bitcast i32 %r to i32
  ret i32 %t
}

; CHECK-LABEL:  define i32 @call_preallocated_musttail_cast(i32* preallocated(i32) %a)
; CHECK:          call void @__tsan_func_exit()
; CHECK-NEXT:     %r = musttail call i32 @preallocated_musttail(i32* preallocated(i32) %a)
; CHECK-NEXT:     %t = bitcast i32 %r to i32
; CHECK-NEXT:     ret i32 %t
