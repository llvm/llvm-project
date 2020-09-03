; RUN: llc -O0 -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-darwin-ios13.0"

declare i8* @objc_msgSend(i8*, i8*, ...)
define void @call_bitcast_ptr_const() {
; CHECK-LABEL: @call_bitcast_ptr_const
; CHECK: bl _objc_msgSend
; CHECK-NOT: blr
entry:
  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, [2 x i32], i32, float)*)(i8* undef, i8* undef, [2 x i32] zeroinitializer, i32 0, float 1.000000e+00)
  ret void
}
