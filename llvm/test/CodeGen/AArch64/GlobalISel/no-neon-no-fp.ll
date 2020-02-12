; RUN: not --crash llc -o - -verify-machineinstrs -global-isel -global-isel-abort=1 -stop-after=legalizer %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-unknown"

; CHECK: unable to legalize instruction: G_STORE %1:_(s128), %0:_(p0) :: (store 16 into %ir.ptr) (in function: foo)
define void @foo(i128 *%ptr) #0 align 2 {
entry:
  store i128 0, i128* %ptr, align 16
  ret void
}

attributes #0 = { "use-soft-float"="false" "target-features"="-fp-armv8,-neon" }

