; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB

define void @t1(i8* %x) {
entry:
; ARM: t1
; THUMB: t1
  br label %L0

L0:
  br label %L1

L1:
  indirectbr i8* %x, [ label %L0, label %L1 ]
; ARM: bx r0
; THUMB: mov pc, r0
}
