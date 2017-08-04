; RUN: llc < %s | FileCheck %s
; RUN: llc -relocation-model=pic < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@cmp8 = external hidden global i8, !absolute_symbol !0
@cmp32 = external hidden global i8, !absolute_symbol !1

declare void @f()

define void @foo8(i64 %val) {
  ; CHECK: cmpq $cmp8@ABS8, %rdi
  %cmp = icmp ule i64 %val, ptrtoint (i8* @cmp8 to i64)
  br i1 %cmp, label %t, label %f

t:
  call void @f()
  ret void

f:
  ret void
}

define void @foo32(i64 %val) {
  ; CHECK: cmpq $cmp32, %rdi
  %cmp = icmp ule i64 %val, ptrtoint (i8* @cmp32 to i64)
  br i1 %cmp, label %t, label %f

t:
  call void @f()
  ret void

f:
  ret void
}

!0 = !{i64 0, i64 128}
!1 = !{i64 0, i64 2147483648}
