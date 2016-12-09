; RUN: llc < %s | FileCheck %s
; RUN: llc -relocation-model=pic < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@bit_mask8 = external hidden global i8, !absolute_symbol !0
@bit_mask32 = external hidden global i8, !absolute_symbol !1
@bit_mask64 = external hidden global i8, !absolute_symbol !2

declare void @f()

define void @foo8(i8* %ptr) {
  %load = load i8, i8* %ptr
  ; CHECK: testb $bit_mask8, (%rdi)
  %and = and i8 %load, ptrtoint (i8* @bit_mask8 to i8)
  %icmp = icmp eq i8 %and, 0
  br i1 %icmp, label %t, label %f

t:
  call void @f()
  ret void

f:
  ret void
}

define void @foo32(i32* %ptr) {
  %load = load i32, i32* %ptr
  ; CHECK: testl $bit_mask32, (%rdi)
  %and = and i32 %load, ptrtoint (i8* @bit_mask32 to i32)
  %icmp = icmp eq i32 %and, 0
  br i1 %icmp, label %t, label %f

t:
  call void @f()
  ret void

f:
  ret void
}

define void @foo64(i64* %ptr) {
  %load = load i64, i64* %ptr
  ; CHECK: movabsq $bit_mask64, %rax
  ; CHECK: testq (%rdi), %rax
  %and = and i64 %load, ptrtoint (i8* @bit_mask64 to i64)
  %icmp = icmp eq i64 %and, 0
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
