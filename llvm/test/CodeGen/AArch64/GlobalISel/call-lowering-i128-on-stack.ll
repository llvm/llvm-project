; RUN: llc -O0 -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

; Check we don't assert when handling an i128 split arg on the stack.
; CHECK-LABEL: fn1
; CHECK: ret
define i32 @fn1(i32 %p1, i128 %p2.coerce, i32 %p3, i32 %p4, i32 %p5, i128 %p6.coerce, i32 %p7) {
entry:
  ret i32 undef
}
