; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

; Ensure we're generating ldp instructions instead of ldr Q.
; CHECK: ldp
; CHECK: stp
define void @f(ptr %p, ptr %q) {
  %addr2 = getelementptr i64, ptr %q, i32 1
  %addr = getelementptr i64, ptr %p, i32 1
  %x = load i64, ptr %p
  %y = load i64, ptr %addr
  store i64 %x, ptr %q
  store i64 %y, ptr %addr2
  ret void
}
