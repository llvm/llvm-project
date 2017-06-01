; RUN: llc -mtriple=x86_64-windows -show-mc-encoding < %s | FileCheck %s

; The Win64 ABI wants tail jmps to use a REX_W prefix so it can distinguish
; in-function jumps from function exiting jumps.

define void @tail_jmp_reg(i32, i32, void ()* %fptr) {
  tail call void () %fptr()
  ret void
}

; Check that we merge the REX prefixes into 0x49 instead of 0x48, 0x41.

; CHECK-LABEL: tail_jmp_reg:
; CHECK: rex64 jmpq *%r8
; CHECK: 	encoding: [0x49,0xff,0xe0]

declare void @tail_tgt()

define void @tail_jmp_imm() {
  tail call void @tail_tgt()
  ret void
}

; CHECK-LABEL: tail_jmp_imm:
; CHECK: jmp tail_tgt

@g_fptr = global void ()* @tail_tgt

define void @tail_jmp_mem() {
  %fptr = load void ()*, void ()** @g_fptr
  tail call void () %fptr()
  ret void
}

; CHECK-LABEL: tail_jmp_mem:
; CHECK: rex64 jmpq *g_fptr(%rip)
