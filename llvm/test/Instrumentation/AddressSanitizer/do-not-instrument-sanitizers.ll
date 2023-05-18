; This test checks that we are not instrumenting sanitizer code.
; RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @__asan_default_options(ptr %a) sanitize_address {
entry:
  %tmp1 = load i32, ptr %a, align 4
  %tmp2 = add i32 %tmp1,  1
  store i32 %tmp2, ptr %a, align 4
  ret void
}

; CHECK-NOT: call void @__asan_report_load

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  ret i32 0
}

; CHECK: declare void @__asan_init()
