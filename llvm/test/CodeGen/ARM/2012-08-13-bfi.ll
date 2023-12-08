; RUN: llc -mcpu=cortex-a8 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "thumb-apple-macosx10.8.0"

; CHECK: foo
; CHECK-NOT: bfi
; CHECK: bx
define i32 @foo(i8 zeroext %i) nounwind uwtable readnone ssp {
  %1 = and i8 %i, 15
  %2 = zext i8 %1 to i32
  %3 = icmp ult i8 %1, 10
  %4 = or i32 %2, 48
  %5 = add nsw i32 %2, 55
  %6 = select i1 %3, i32 %4, i32 %5
  ret i32 %6
}
