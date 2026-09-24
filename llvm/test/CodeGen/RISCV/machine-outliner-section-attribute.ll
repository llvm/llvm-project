; NOTE: This test is derived from the following C source. It uses only
; parameter arithmetic so the repeated sequence is self-contained.
;
; #define BODY(N) do { a ^= b; b += a; a = (a << 3) | (a >> 29); \
;                       a ^= b; return a + N; } while (0)
; __attribute__((noinline, section(".sec_shared")))
; unsigned shared_0(unsigned a, unsigned b) { BODY(1); }
; __attribute__((noinline, section(".sec_shared")))
; unsigned shared_1(unsigned a, unsigned b) { BODY(2); }
; __attribute__((noinline, section(".sec_shared")))
; unsigned shared_2(unsigned a, unsigned b) { BODY(3); }

; RUN: llc -mtriple=riscv32 -enable-machine-outliner=always -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple=riscv32 -enable-machine-outliner=always --function-sections -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple=riscv64 -enable-machine-outliner=always -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV64
; RUN: llc -mtriple=riscv64 -enable-machine-outliner=always --function-sections -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV64

; CHECK: .section .sec_shared,"ax",@progbits

define i32 @shared_0(i32 %a, i32 %b) noinline nounwind section ".sec_shared" {
; CHECK-LABEL: shared_0:
; CHECK: call t0, OUTLINED_FUNCTION_0
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 1
  ret i32 %result
}

define i32 @shared_1(i32 %a, i32 %b) noinline nounwind section ".sec_shared" {
; CHECK-LABEL: shared_1:
; CHECK: call t0, OUTLINED_FUNCTION_0
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 2
  ret i32 %result
}

define i32 @shared_2(i32 %a, i32 %b) noinline nounwind section ".sec_shared" {
; CHECK-LABEL: shared_2:
; CHECK: call t0, OUTLINED_FUNCTION_0
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 3
  ret i32 %result
}

; The outlined function inherits the input section of its parents.
; CHECK: OUTLINED_FUNCTION_0:
; CHECK: xor a0, a0, a1
; RV32-NEXT: srli a2, a0, 29
; RV64-NEXT: srliw a2, a0, 29
; CHECK-NEXT: slli a3, a0, 3
; CHECK-NEXT: add a0, a1, a0
; CHECK-NEXT: or a2, a3, a2
; CHECK-NEXT: xor a0, a2, a0
; CHECK-NEXT: jr t0
