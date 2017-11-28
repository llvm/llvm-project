; REQUIRES: asserts
; RUN: llc -verify-machineinstrs -mtriple=x86_64-apple-darwin -debug -o /dev/null < %s 2>&1 | FileCheck %s

; We need to make sure that rematerialization into a physical register marks the
; super- or sub-register as dead after this rematerialization since only the
; original register is actually used later. Largely irrelevant for a trivial
; example like this, since eax is never used again, but easy to test.

define i8 @test_remat() {
  ret i8 0
; CHECK: REGISTER COALESCING
; CHECK: Remat: %eax<def,dead> = MOV32r0 %eflags<imp-def,dead>, %al<imp-def>
}

; On the other hand, if it's already the correct width, we really shouldn't be
; marking the definition register as dead.

define i32 @test_remat32() {
  ret i32 0
; CHECK: REGISTER COALESCING
; CHECK: Remat: %eax<def> = MOV32r0 %eflags<imp-def,dead>
}

