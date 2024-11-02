; Test -sanitizer-coverage-control-flow.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-control-flow -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo(ptr %a) sanitize_address {
entry:
  %tobool = icmp eq i32* %a, null
  br i1 %tobool, label %if.end, label %if.then

  if.then:                                          ; preds = %entry
  store i32 0, i32* %a, align 4
  call void @foo(i32* %a)
  br label %if.end

  if.end:                                           ; preds = %entry, %if.then
  ret void
}

; CHECK: private constant [17 x ptr] [{{.*}}@foo{{.*}}blockaddress{{.*}}blockaddress{{.*}}blockaddress{{.*}}blockaddress{{.*}}blockaddress{{.*}}blockaddress{{.*}}blockaddress{{.*}}@foo{{.*}}null{{.*}}null], section "__sancov_cfs", comdat($foo), align 8
; CHECK:      @__start___sancov_cfs = extern_weak hidden global
; CHECK-NEXT: @__stop___sancov_cfs = extern_weak hidden global
