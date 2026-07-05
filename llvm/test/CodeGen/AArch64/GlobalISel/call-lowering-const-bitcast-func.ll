; RUN: llc -O0 -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-darwin-ios13.0"

declare ptr @objc_msgSend(ptr, ptr, ...)
define void @call_bitcast_ptr_const() {
; CHECK-LABEL: @call_bitcast_ptr_const
; CHECK: bl _objc_msgSend
; CHECK-NOT: blr
entry:
  call void @objc_msgSend(ptr undef, ptr undef, [2 x i32] zeroinitializer, i32 0, float 1.000000e+00)
  ret void
}
