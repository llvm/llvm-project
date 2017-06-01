; Check the "no-realign-stack" function attribute. We should get a warning.

; RUN: llc < %s -mtriple=s390x-linux-gnu -debug-only=codegen 2>&1 | \
; RUN:   FileCheck %s
; REQUIRES: asserts

define void @f6() "no-realign-stack" {
  %x = alloca i64, i64 1, align 128
  store volatile i64 10, i64* %x, align 128
  ret void
}

; CHECK: Warning: requested alignment 128 exceeds the stack alignment 8
; CHECK-NOT: nill
