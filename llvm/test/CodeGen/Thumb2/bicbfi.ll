; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7--linux-gnueabihf"

; CHECK-LABEL: f:
; CHECK: bic
define void @f(ptr nocapture %b, ptr nocapture %c, i32 %a) {
  %1 = and i32 %a, -4096
  store i32 %1, ptr %c, align 4
  %2 = and i32 %a, 4095
  %3 = or i32 %2, 4096
  %4 = load i32, ptr %b, align 4
  %5 = add nsw i32 %4, %3
  store i32 %5, ptr %b, align 4
  ret void
}
