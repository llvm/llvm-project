; RUN: opt < %s -passes=rtsan -S | FileCheck %s

define void @nosanitized_function() #0 {
  %1 = alloca ptr, align 8
  %2 = call ptr @malloc(i64 noundef 2) #3
  store ptr %2, ptr %1, align 8
  ret void
}

declare ptr @malloc(i64 noundef) #1

define noundef i32 @main() #2 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  call void @nosanitized_function() #4
  ret i32 0
}

attributes #0 = { nosanitize_realtime }

; CHECK-LABEL: @nosanitized_function()
; CHECK-NEXT: call{{.*}}@__rtsan_off

; CHECK: call{{.*}}@__rtsan_on
; CHECK-NEXT: ret{{.*}}void
