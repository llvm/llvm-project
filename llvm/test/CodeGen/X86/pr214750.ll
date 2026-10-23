; RUN: llc -O0 -mtriple=x86_64 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=x86_64 -verify-machineinstrs < %s | FileCheck %s

; PR214750: a static alloca of 0xFFFFFFFFFFFFFFFF bytes collided with the
; ~0ULL size sentinel MachineFrameInfo used for dead stack objects, so PEI
; never assigned it an offset and eliminateFrameIndex asserted. Just check
; that this compiles.

define void @f() nounwind {
; CHECK-LABEL: f:
; CHECK: retq
  %a = alloca ptr, align 8
  %b = alloca i8, i64 -1, align 16
  store ptr %b, ptr %a, align 8
  ret void
}

define ptr @escape() nounwind {
; CHECK-LABEL: escape:
; CHECK: retq
  %p = alloca i8, i64 -1, align 16
  ret ptr %p
}
