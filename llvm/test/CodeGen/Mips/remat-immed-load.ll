; RUN: llc -mtriple=mipsel < %s | FileCheck %s -check-prefix=32
; RUN: llc -mtriple=mips64el -mcpu=mips4 -target-abi=n64 < %s | FileCheck %s -check-prefix=64
; RUN: llc -mtriple=mips64el -mcpu=mips64 -target-abi=n64 < %s | FileCheck %s -check-prefix=64

define void @f0() nounwind {
entry:
; 32:  addiu $4, $zero, 1
; 32:  addiu $4, $zero, 1

  tail call void @foo1(i32 1) nounwind
  tail call void @foo1(i32 1) nounwind
  ret void
}

declare void @foo1(i32)

define void @f3() nounwind {
entry:
; 64:  daddiu $4, $zero, 1
; 64:  daddiu $4, $zero, 1

  tail call void @foo2(i64 1) nounwind
  tail call void @foo2(i64 1) nounwind
  ret void
}

declare void @foo2(i64)

define void @f5() nounwind {
entry:
; 32:  lui $4, 1
; 32:  lui $4, 1

  tail call void @f6(i32 65536) nounwind
  tail call void @f6(i32 65536) nounwind
  ret void
}

declare void @f6(i32)

define void @f7() nounwind {
entry:
; 64:  lui $4, 1
; 64:  lui $4, 1

  tail call void @f8(i64 65536) nounwind
  tail call void @f8(i64 65536) nounwind
  ret void
}

declare void @f8(i64)

