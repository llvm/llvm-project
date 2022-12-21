; RUN: llc < %s -verify-machineinstrs

; Make sure we don't end up with a CBNZ of an undef v-/phys-reg.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

declare void @bar(ptr)

define void @foo(ptr %m, i32 %off0) {
.thread1653:
  br i1 undef, label %0, label %.thread1880

  %1 = icmp eq i32 undef, 0
  %.not = xor i1 %1, true
  %brmerge = or i1 %.not, undef
  br i1 %brmerge, label %.thread1880, label %.thread1705

.thread1705:
  ret void

.thread1880:
  %m1652.ph = phi ptr [ %m, %0 ], [ null, %.thread1653 ]
  call void @bar(ptr %m1652.ph)
  ret void
}
