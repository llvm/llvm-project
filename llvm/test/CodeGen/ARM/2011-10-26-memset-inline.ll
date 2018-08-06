; Make sure short memsets on ARM lower to stores, even when optimizing for size.
; RUN: llc -mattr=+strict-align < %s | FileCheck %s -check-prefix=CHECK-GENERIC
; RUN: llc -mcpu=cortex-a8 < %s | FileCheck %s -check-prefix=CHECK-UNALIGNED

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios5.0.0"

; CHECK-GENERIC:      strb
; CHECK-GENERIC-NEXT: strb
; CHECK-GENERIC-NEXT: strb
; CHECK-GENERIC-NEXT: strb
; CHECK-GENERIC-NEXT: strb
; CHECK-UNALIGNED:    strb
; CHECK-UNALIGNED:    str
define void @foo(i8* nocapture %c) nounwind optsize {
entry:
  call void @llvm.memset.p0i8.i64(i8* %c, i8 -1, i64 5, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
