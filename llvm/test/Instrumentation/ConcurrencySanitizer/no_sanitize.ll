; RUN: opt < %s -passes='function(csan)' -S -mtriple=x86_64-unknown-linux-gnu | FileCheck %s
; RUN: opt < %s -passes='function(csan)' -S -mtriple=amdgcn-amd-amdhsa | FileCheck %s

define i32 @unattributed(ptr %a) {
entry:
  %v = load i32, ptr %a, align 4
  ret i32 %v
}
; CHECK-LABEL: @unattributed(
; CHECK-NEXT: entry:
; CHECK-NEXT: %v = load i32, ptr %a, align 4
; CHECK-NEXT: ret i32 %v

define i32 @thread_only(ptr %a) sanitize_thread {
entry:
  %v = load i32, ptr %a, align 4
  ret i32 %v
}
; CHECK-LABEL: @thread_only(
; CHECK-NEXT: entry:
; CHECK-NEXT: %v = load i32, ptr %a, align 4
; CHECK-NEXT: ret i32 %v

define i32 @instrumented(ptr %a) sanitize_concurrency {
entry:
  %v = load i32, ptr %a, align 4
  ret i32 %v
}
; CHECK-LABEL: @instrumented(
; CHECK: call void @__csan_read4(ptr %a, i32 0)

define i32 @disabled(ptr %a) sanitize_concurrency disable_sanitizer_instrumentation {
entry:
  %v = load i32, ptr %a, align 4
  ret i32 %v
}
; CHECK-LABEL: @disabled(
; CHECK-NEXT: entry:
; CHECK-NEXT: %v = load i32, ptr %a, align 4
; CHECK-NEXT: ret i32 %v

declare void @callee()

define i32 @checking_suppressed(ptr %a) sanitize_concurrency "sanitize_concurrency_no_checking_at_run_time" {
entry:
  %v = load i32, ptr %a, align 4
  call void @callee()
  ret i32 %v
}
; CHECK-LABEL: @checking_suppressed(
; CHECK: call void @__csan_func_entry
; CHECK: call void @__csan_ignore_thread_begin()
; CHECK-NOT: call void @__csan_read
; CHECK: %v = load i32, ptr %a, align 4
; CHECK: {{call|invoke}} void @callee()
; CHECK: call void @__csan_ignore_thread_end()
; CHECK: call void @__csan_func_exit()
