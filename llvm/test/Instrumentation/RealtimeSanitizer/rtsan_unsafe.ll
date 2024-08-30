; RUN: opt < %s -passes=rtsan -S | FileCheck %s

define void @blocking_function() #0 {
  ret void
}

define noundef i32 @main() #2 {
  call void @blocking_function() #4
  ret i32 0
}

attributes #0 = { mustprogress noinline sanitize_realtime_unsafe optnone ssp uwtable(sync) }

; RealtimeSanitizer pass should insert __rtsan_expect_not_realtime at function entrypoint
; CHECK-LABEL: @blocking_function()
; CHECK-NEXT: call{{.*}}@__rtsan_expect_not_realtime({{ptr .*}})
