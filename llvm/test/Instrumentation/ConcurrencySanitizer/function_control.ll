; RUN: opt < %s -passes='function(csan)' -S | FileCheck %s --check-prefixes=CHECK,EH
; RUN: opt < %s -passes='function(csan)' -csan-handle-cxx-exceptions=0 -S | FileCheck %s --check-prefixes=CHECK,NOEH
; RUN: opt < %s -passes='function(csan)' -csan-instrument-func-entry-exit=0 -S | FileCheck %s --check-prefix=NOENTRY

declare void @may_throw()

; Entry and exit callbacks must remain balanced when a call unwinds.
define void @exception_exit() sanitize_concurrency {
entry:
  call void @may_throw()
  ret void
}
; EH-LABEL: @exception_exit(
; EH: invoke void @may_throw()
; EH: csan_cleanup:
; EH: call void @__csan_func_exit()
; EH-NEXT: resume
; NOEH-LABEL: @exception_exit(
; NOEH: call void @may_throw()
; NOEH: call void @__csan_func_exit()

define void @unattributed_call() {
entry:
  call void @may_throw()
  ret void
}
; CHECK-LABEL: @unattributed_call(
; CHECK: call void @__csan_func_entry
; CHECK: {{call|invoke}} void @may_throw()
; CHECK: call void @__csan_func_exit()

define i32 @musttail_callee(ptr %p) sanitize_concurrency {
entry:
  %v = load i32, ptr %p
  ret i32 %v
}

; A musttail call must remain immediately before its return.
define i32 @musttail_caller(ptr %p) sanitize_concurrency {
entry:
  %v = musttail call i32 @musttail_callee(ptr %p)
  ret i32 %v
}
; CHECK-LABEL: @musttail_caller(
; CHECK: call void @__csan_func_exit()
; CHECK-NEXT: %v = musttail call i32 @musttail_callee(ptr %p)
; CHECK-NEXT: ret i32 %v

define void @ignored() "sanitize_concurrency_no_checking_at_run_time" {
entry:
  call void @may_throw()
  ret void
}
; NOENTRY-LABEL: @ignored(
; NOENTRY-NOT: __csan_func_
; NOENTRY: call void @__csan_ignore_thread_begin()
; NOENTRY: invoke void @may_throw()
; NOENTRY: call void @__csan_ignore_thread_end()
; NOENTRY: ret void
