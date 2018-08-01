; RUN: llc -O0 -o - %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; CHECK-LABEL: testfn
; CHECK: ret
define void @testfn() {
start:
  br label %bb2

bb1:
  store i8 %0, i8* undef, align 4
  ret void

bb2:
  %0 = extractvalue { i32, i8 } undef, 1
  br label %bb1
}

