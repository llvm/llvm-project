; RUN: opt < %s -passes=rtsan -S | FileCheck %s

define void @_Z17blocking_functionv() #0 {
  ret void
}

define noundef i32 @main() #2 {
  call void @_Z17blocking_functionv() #4
  ret i32 0
}

attributes #0 = { mustprogress noinline sanitize_realtime_unsafe optnone ssp uwtable(sync) }

; RealtimeSanitizer pass should create the demangled function name as a global string and,
; at the function entrypoint, pass it as an argument to the rtsan notify method
; CHECK: [[GLOBAL_STR:@[a-zA-Z0-9\.]+]]
; CHECK-SAME: c"blocking_function()\00"
; CHECK-LABEL: @_Z17blocking_functionv()
; CHECK-NEXT: call{{.*}}@__rtsan_notify_blocking_call(ptr{{.*}}[[GLOBAL_STR]])
