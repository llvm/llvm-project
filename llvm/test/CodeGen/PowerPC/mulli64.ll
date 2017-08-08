; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i64 @foo(i64 %a) #0 {
entry:
  %mul = mul nsw i64 %a, 3
  ret i64 %mul
}

; CHECK-LABEL: @foo
; CHECK: mulli 3, 3, 3
; CHECK: blr

attributes #0 = { nounwind readnone }

