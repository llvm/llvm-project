; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=2 -sanitizer-coverage-trace-pc-entry-exit -S | FileCheck %s --check-prefix=CHECK_EE_ONLY
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=2 -sanitizer-coverage-trace-pc-entry-exit -sanitizer-coverage-trace-pc -S | FileCheck %s --check-prefix=CHECK_EE_COMBINED

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo(ptr %a) #0 {
entry:
  %tobool = icmp eq ptr %a, null
  br i1 %tobool, label %if.end, label %if.then

  if.then:                                          ; preds = %entry
  store i32 0, ptr %a, align 4
  br label %if.end

  if.end:                                           ; preds = %entry, %if.then
  ret void
}
attributes #0 = { nounwind }

; CHECK_EE_ONLY: call void @__sanitizer_cov_trace_pc_entry
; CHECK_EE_ONLY-NOT: call void @__sanitizer_cov_trace_pc
; CHECK_EE_ONLY: call void @__sanitizer_cov_trace_pc_exit

; CHECK_EE_COMBINED: call void @__sanitizer_cov_trace_pc_entry
; CHECK_EE_COMBINED: call void @__sanitizer_cov_trace_pc
; CHECK_EE_COMBINED: call void @__sanitizer_cov_trace_pc_exit
