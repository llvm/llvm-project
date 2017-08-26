; Test -sanitizer-coverage-pc-table=1
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard       -sanitizer-coverage-pc-table=1 -S | FileCheck %s
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-inline-8bit-counters -sanitizer-coverage-pc-table=1 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo(i32* %a) sanitize_address {
entry:
  %tobool = icmp eq i32* %a, null
  br i1 %tobool, label %if.end, label %if.then

  if.then:                                          ; preds = %entry
  store i32 0, i32* %a, align 4
  br label %if.end

  if.end:                                           ; preds = %entry, %if.then
  ret void
}

; CHECK: private constant [6 x i64*] [{{.*}}@foo{{.*}}blockaddress{{.*}}blockaddress{{.*}}], section "__sancov_pcs", align 8
; CHECK: define internal void @sancov.module_ctor
; CHECK: call void @__sanitizer_cov
; CHECK: call void @__sanitizer_cov_pcs_init
