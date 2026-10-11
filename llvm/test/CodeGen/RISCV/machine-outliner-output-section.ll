; NOTE: This test is derived from the following C source. With LTO + a linker
; script the linker records the output section each function is mapped to as the
; "linker_output_section" attribute.
;
; same_out_0/1 are in different input sections but map to one output section, so
; they are outlined together. diff_out maps elsewhere and is not grouped with
; them. no_attr_0/1 carry no attribute and fall back to their shared input
; section.
;
; #define BODY(N) do { a ^= b; b += a; a = (a << 3) | (a >> 29); \
;                       a ^= b; return a + N; } while (0)
; #define SEC(S, N, V) __attribute__((noinline, section(S))) \
;   unsigned N(unsigned a, unsigned b) { BODY(V); }
; SEC(".text.same_out_0", same_out_0, 1) SEC(".text.same_out_1", same_out_1, 2)
; SEC(".text.diff_out", diff_out, 3)
; SEC(".text.shared", no_attr_0, 4) SEC(".text.shared", no_attr_1, 5)

; RUN: llc -mtriple=riscv32 -enable-machine-outliner=always -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -enable-machine-outliner=always -verify-machineinstrs < %s | FileCheck %s

; The two functions sharing an output section are outlined together even though
; their input sections differ.
define i32 @same_out_0(i32 %a, i32 %b) noinline nounwind #0 section ".text.same_out_0" {
; CHECK: .section .text.same_out_0,"ax",@progbits
; CHECK-LABEL: same_out_0:
; CHECK: call t0, OUTLINED_FUNCTION_[[HOT:[0-9_]+]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 1
  ret i32 %result
}

define i32 @same_out_1(i32 %a, i32 %b) noinline nounwind #0 section ".text.same_out_1" {
; CHECK: .section .text.same_out_1,"ax",@progbits
; CHECK-LABEL: same_out_1:
; CHECK: call t0, OUTLINED_FUNCTION_[[HOT]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 2
  ret i32 %result
}

; A different output section must not be grouped with the pair above.
define i32 @diff_out(i32 %a, i32 %b) noinline nounwind #1 section ".text.diff_out" {
; CHECK: .section .text.diff_out,"ax",@progbits
; CHECK-LABEL: diff_out:
; CHECK-NOT: call t0, OUTLINED_FUNCTION_
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 3
  ret i32 %result
}

; No attribute: the input section is used, and these two share it.
define i32 @no_attr_0(i32 %a, i32 %b) noinline nounwind section ".text.shared" {
; CHECK: .section .text.shared,"ax",@progbits
; CHECK-LABEL: no_attr_0:
; CHECK: call t0, OUTLINED_FUNCTION_[[SHARED:[0-9_]+]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 4
  ret i32 %result
}

define i32 @no_attr_1(i32 %a, i32 %b) noinline nounwind section ".text.shared" {
; CHECK-LABEL: no_attr_1:
; CHECK: call t0, OUTLINED_FUNCTION_[[SHARED]]
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 5
  ret i32 %result
}

attributes #0 = { "linker_output_section"="out_hot" }
attributes #1 = { "linker_output_section"="out_cold" }

; Each outlined function is emitted into the input section of its candidates.
; CHECK: .section .text.same_out_0,"ax",@progbits
; CHECK: OUTLINED_FUNCTION_[[HOT]]:
; CHECK: .section .text.shared,"ax",@progbits
; CHECK: OUTLINED_FUNCTION_[[SHARED]]:
