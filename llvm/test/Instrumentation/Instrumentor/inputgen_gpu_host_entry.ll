; Verify entry generation reports unsupported host targets without changing IR.
; RUN: opt < %s -passes=inputgen-gpu -inputgen-gpu-entry-function=vvv_foo -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: inputgen entry kernels are not supported for target 'x86_64-unknown-linux-gnu'
; CHECK-NOT: __ig_post_load
; CHECK-NOT: __ig_entry
; CHECK-LABEL: define i32 @vvv_foo(
; CHECK: load i32, ptr %a, align 4

define i32 @vvv_foo(ptr noundef %a) {
entry:
  %v = load i32, ptr %a, align 4
  ret i32 %v
}
