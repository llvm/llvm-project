; RUN: opt < %s -passes=rtsan -S | FileCheck %s

define void @violation() #0 {
  %1 = alloca ptr, align 8
  %2 = call ptr @malloc(i64 noundef 2) #3
  store ptr %2, ptr %1, align 8
  ret void
}

declare ptr @malloc(i64 noundef) #1

define noundef i32 @main() #2 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  call void @violation() #4
  ret i32 0
}

attributes #0 = { mustprogress noinline sanitize_realtime optnone ssp uwtable(sync) }

; RealtimeSanitizer pass should insert __rtsan_realtime_enter right after function definition
; CHECK-LABEL: @violation()
; CHECK-NEXT: call{{.*}}@__rtsan_realtime_enter

; RealtimeSanitizer pass should insert __rtsan_realtime_exit right before function return
; CHECK: call{{.*}}@__rtsan_realtime_exit
; CHECK-NEXT: ret{{.*}}void
