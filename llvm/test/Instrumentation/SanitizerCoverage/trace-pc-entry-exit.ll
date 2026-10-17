; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=2 -sanitizer-coverage-trace-pc-entry-exit -S | FileCheck %s --check-prefix=CHECK
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=2 -sanitizer-coverage-trace-pc-entry-exit -sanitizer-coverage-trace-pc -S | FileCheck %s --check-prefix=CHECK,CHECK_BBCOV

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @callee_a(i32)
declare i32 @callee_b()

define i32 @func_multi_return(i32 noundef %a) #0 {
entry:
  %tobool.not = icmp eq i32 %a, 0
  br i1 %tobool.not, label %if.else, label %if.then

if.then:
  %call = call i32 @callee_a(i32 noundef %a)
  ret i32 %call

if.else:
  %call1 = call i32 @callee_b()
  ret i32 %call1
}

; CHECK: define i32 @func_multi_return
; CHECK-NOT: call void @__sanitizer_cov_trace_pc()
; CHECK: call void @__sanitizer_cov_trace_pc_entry()
; CHECK-NOT: call void @__sanitizer_cov_trace_pc()
; CHECK: br
; CHECK_BBCOV: call void @__sanitizer_cov_trace_pc()
; CHECK: call i32 @callee_a
; CHECK: notail call void @__sanitizer_cov_trace_pc_exit()
; CHECK: ret
; CHECK_BBCOV: call void @__sanitizer_cov_trace_pc()
; CHECK: call i32 @callee_b
; CHECK: notail call void @__sanitizer_cov_trace_pc_exit()
; CHECK: ret
; CHECK: }


define i32 @func_musttail(i32 noundef %a) #0 {
  %call = musttail call i32 @callee_a(i32 noundef %a)
  ret i32 %call
}

; CHECK: define i32 @func_musttail
; CHECK: call void @__sanitizer_cov_trace_pc_entry()
; CHECK: notail call void @__sanitizer_cov_trace_pc_exit()
; CHECK: musttail call i32 @callee_a
; CHECK: ret
; CHECK: }

attributes #0 = { nounwind }
