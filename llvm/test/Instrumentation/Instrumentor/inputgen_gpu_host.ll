; Verify the GPU policy leaves a host-targeted function unchanged.
; RUN: opt < %s -passes=inputgen-gpu -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The inputgen-gpu pass is GPU-only, so host IR should remain unchanged.
; CHECK-LABEL: define i32 @vvv_foo(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %v = load i32, ptr %a, align 4
; CHECK-NEXT:   ret i32 %v
; CHECK-NEXT: }

define i32 @vvv_foo(ptr noundef %a) {
entry:
  %v = load i32, ptr %a, align 4
  ret i32 %v
}
