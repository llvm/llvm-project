; RUN: opt -S -mtriple=riscv32 -mattr=+a -passes='require<libcall-lowering-info>,atomic-expand' %s | FileCheck %s

; Check that widenPartwordAtomicRMW preserves the volatile bit.

define i8 @test_atomicrmw_and_i8_volatile(ptr %p) {
; CHECK-LABEL: @test_atomicrmw_and_i8_volatile(
; CHECK:         atomicrmw volatile and ptr {{.*}}, i32 {{.*}} seq_cst, align 4
  %r = atomicrmw volatile and ptr %p, i8 15 seq_cst, align 1
  ret i8 %r
}

define i8 @test_atomicrmw_or_i8_volatile(ptr %p) {
; CHECK-LABEL: @test_atomicrmw_or_i8_volatile(
; CHECK:         atomicrmw volatile or ptr {{.*}}, i32 {{.*}} seq_cst, align 4
  %r = atomicrmw volatile or ptr %p, i8 15 seq_cst, align 1
  ret i8 %r
}

define i8 @test_atomicrmw_xor_i8_volatile(ptr %p) {
; CHECK-LABEL: @test_atomicrmw_xor_i8_volatile(
; CHECK:         atomicrmw volatile xor ptr {{.*}}, i32 {{.*}} seq_cst, align 4
  %r = atomicrmw volatile xor ptr %p, i8 15 seq_cst, align 1
  ret i8 %r
}

; Ensure the non-volatile case is not promoted to volatile.
define i8 @test_atomicrmw_and_i8_nonvolatile(ptr %p) {
; CHECK-LABEL: @test_atomicrmw_and_i8_nonvolatile(
; CHECK-NOT:     atomicrmw volatile
; CHECK:         atomicrmw and ptr {{.*}}, i32 {{.*}} seq_cst, align 4
  %r = atomicrmw and ptr %p, i8 15 seq_cst, align 1
  ret i8 %r
}

; Also exercise i16 widening.
define i16 @test_atomicrmw_and_i16_volatile(ptr %p) {
; CHECK-LABEL: @test_atomicrmw_and_i16_volatile(
; CHECK:         atomicrmw volatile and ptr {{.*}}, i32 {{.*}} seq_cst, align 4
  %r = atomicrmw volatile and ptr %p, i16 15 seq_cst, align 2
  ret i16 %r
}
