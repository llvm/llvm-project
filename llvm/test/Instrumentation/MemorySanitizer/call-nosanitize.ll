; Verify that calls with !nosanitize are not instrumented by MSan.
; RUN: opt < %s -msan -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @bar(i32 %x)

define void @foo() {
  call void @bar(i32 7), !nosanitize !{}
  ret void
}

; CHECK-LABEL: define void @foo
; CHECK-NOT: store i{{[0-9]+}} 0, {{.*}} @__msan_param_tls
; CHECK: call void @bar
; CHECK: ret void
