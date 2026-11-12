; RUN: llc < %s | FileCheck %s
; RUN: llc -relocation-model=pic < %s | FileCheck %s

; Regression test for PR38200

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@bit_mask8 = external hidden global i8, !absolute_symbol !0

declare void @f()

define void @foo8(ptr %ptr) noinline optnone {
  %load = load i8, ptr %ptr
  ; CHECK: movl $bit_mask8, %ecx
  %and = and i8 %load, ptrtoint (ptr @bit_mask8 to i8)
  %icmp = icmp eq i8 %and, 0
  br i1 %icmp, label %t, label %f

t:
  call void @f()
  ret void

f:
  ret void
}

!0 = !{i64 0, i64 256}
