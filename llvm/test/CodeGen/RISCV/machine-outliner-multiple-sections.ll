; NOTE: This test is derived from the following C source. It exercises one
; repeated sequence across unsectioned functions and three explicit sections.
;
; All four input-section buckets are outlined in one pass.
;
; #define BODY(N) do { a ^= b; b += a; a = (a << 3) | (a >> 29); \
;                       a ^= b; return a + N; } while (0)
; #define SEC(N, V) __attribute__((noinline, section(".sec_shared"))) \
;   unsigned N(unsigned a, unsigned b) { BODY(V); }
; #define SEC2(S, N, V) __attribute__((noinline, section(S))) \
;   unsigned N(unsigned a, unsigned b) { BODY(V); }
; #define PLAIN(N, V) __attribute__((noinline)) \
;   unsigned N(unsigned a, unsigned b) { BODY(V); }
; SEC(shared_0, 1) SEC(shared_1, 2) SEC(shared_2, 3)
; PLAIN(plain_0, 4) PLAIN(plain_1, 5) PLAIN(plain_2, 6)
; SEC2(".sec_a", a_0, 7) SEC2(".sec_a", a_1, 8)
; SEC2(".sec_b", b_0, 9) SEC2(".sec_b", b_1, 10)

; RUN: llc -mtriple=riscv32 -enable-machine-outliner=always -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,NOFS
; RUN: llc -mtriple=riscv32 -enable-machine-outliner=always --function-sections -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,FS
; RUN: llc -mtriple=riscv64 -enable-machine-outliner=always -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,NOFS
; RUN: llc -mtriple=riscv64 -enable-machine-outliner=always --function-sections -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,FS

; CHECK: .section .sec_shared,"ax",@progbits

define i32 @shared_0(i32 %a, i32 %b) noinline nounwind section ".sec_shared" {
; CHECK-LABEL: shared_0:
; CHECK: call t0, OUTLINED_FUNCTION_[[SHARED:[0-9_]+]]
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
; CHECK: call t0, OUTLINED_FUNCTION_[[SHARED]]
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
; CHECK: call t0, OUTLINED_FUNCTION_[[SHARED]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 3
  ret i32 %result
}

define i32 @plain_0(i32 %a, i32 %b) noinline nounwind {
; NOFS: .text{{$}}
; FS: .section .text.plain_0,"ax",@progbits
; CHECK-LABEL: plain_0:
; CHECK: call t0, OUTLINED_FUNCTION_[[PLAIN:[0-9_]+]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 4
  ret i32 %result
}

define i32 @plain_1(i32 %a, i32 %b) noinline nounwind {
; FS: .section .text.plain_1,"ax",@progbits
; CHECK-LABEL: plain_1:
; CHECK: call t0, OUTLINED_FUNCTION_[[PLAIN]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 5
  ret i32 %result
}

define i32 @plain_2(i32 %a, i32 %b) noinline nounwind {
; FS: .section .text.plain_2,"ax",@progbits
; CHECK-LABEL: plain_2:
; CHECK: call t0, OUTLINED_FUNCTION_[[PLAIN]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 6
  ret i32 %result
}

define i32 @a_0(i32 %a, i32 %b) noinline nounwind section ".sec_a" {
; CHECK-LABEL: a_0:
; CHECK: call t0, OUTLINED_FUNCTION_[[A:[0-9_]+]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 7
  ret i32 %result
}

define i32 @a_1(i32 %a, i32 %b) noinline nounwind section ".sec_a" {
; CHECK-LABEL: a_1:
; CHECK: call t0, OUTLINED_FUNCTION_[[A]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 8
  ret i32 %result
}

define i32 @b_0(i32 %a, i32 %b) noinline nounwind section ".sec_b" {
; CHECK-LABEL: b_0:
; CHECK: call t0, OUTLINED_FUNCTION_[[B:[0-9_]+]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 9
  ret i32 %result
}

define i32 @b_1(i32 %a, i32 %b) noinline nounwind section ".sec_b" {
; CHECK-LABEL: b_1:
; CHECK: call t0, OUTLINED_FUNCTION_[[B]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 10
  ret i32 %result
}

; All four section buckets were outlined in the same pass.
; CHECK: OUTLINED_FUNCTION_[[SHARED]]:
; NOFS: .text{{$}}
; NOFS: OUTLINED_FUNCTION_[[PLAIN]]:
; FS: .section .text.OUTLINED_FUNCTION_[[PLAIN]],"ax",@progbits
; FS: OUTLINED_FUNCTION_[[PLAIN]]:
; CHECK: .section .sec_a,"ax",@progbits
; CHECK: OUTLINED_FUNCTION_[[A]]:
; CHECK: .section .sec_b,"ax",@progbits
; CHECK: OUTLINED_FUNCTION_[[B]]:
