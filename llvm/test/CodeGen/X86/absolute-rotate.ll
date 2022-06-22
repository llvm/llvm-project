; RUN: llc < %s | FileCheck %s
; RUN: llc -relocation-model=pic < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@align = external hidden global i8, !absolute_symbol !0

declare void @f()

define void @foo(i64 %val) {
  %shr = lshr i64 %val, zext (i8 ptrtoint (ptr @align to i8) to i64)
  %shl = shl i64 %val, zext (i8 sub (i8 64, i8 ptrtoint (ptr @align to i8)) to i64)
  ; CHECK: rorq $align@ABS8, %rdi
  %ror = or i64 %shr, %shl
  %cmp = icmp ult i64 %ror, 109
  br i1 %cmp, label %t, label %f

t:
  call void @f()
  ret void

f:
  ret void
}

define void @bar(i64 %val) {
  %shr = shl i64 %val, zext (i8 ptrtoint (ptr @align to i8) to i64)
  %shl = lshr i64 %val, zext (i8 sub (i8 64, i8 ptrtoint (ptr @align to i8)) to i64)
  ; CHECK: rolq $align@ABS8, %rdi
  %rol = or i64 %shr, %shl
  %cmp = icmp ult i64 %rol, 109
  br i1 %cmp, label %t, label %f

t:
  call void @f()
  ret void

f:
  ret void
}

!0 = !{i64 0, i64 128}
