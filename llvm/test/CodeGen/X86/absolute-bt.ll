; RUN: llc < %s | FileCheck %s
; RUN: llc -relocation-model=pic < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@bit_mask8 = external hidden global i8, !absolute_symbol !0
@bit_mask32 = external hidden global i8, !absolute_symbol !1
@bit_mask64 = external hidden global i8, !absolute_symbol !2

declare void @f()

define void @foo32(ptr %ptr) {
  %load = load i32, ptr %ptr
  %and = and i32 %load, 31
  %shl = shl i32 1, %and
  %and2 = and i32 %shl, ptrtoint (ptr @bit_mask32 to i32)
  ; CHECK: movl $bit_mask32, %eax
  ; CHECK: btl %ecx, %eax
  %icmp = icmp eq i32 %and2, 0
  br i1 %icmp, label %t, label %f

t:
  call void @f()
  ret void

f:
  ret void
}

define void @foo64(ptr %ptr) {
  %load = load i64, ptr %ptr
  %and = and i64 %load, 63
  %shl = shl i64 1, %and
  %and2 = and i64 %shl, ptrtoint (ptr @bit_mask64 to i64)
  ; CHECK: movabsq $bit_mask64, %rax
  ; CHECK: btq %rcx, %rax
  %icmp = icmp eq i64 %and2, 0
  br i1 %icmp, label %t, label %f

t:
  call void @f()
  ret void

f:
  ret void
}

!0 = !{i64 0, i64 256}
!1 = !{i64 0, i64 4294967296}
!2 = !{i64 -1, i64 -1}
