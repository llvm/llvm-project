; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s | FileCheck %s

; The asm has two frame index operands and clobbers nearly every register. With
; a stack too large for 12-bit offsets, both addresses need a scratch register
; at the same time, so two emergency spill slots are needed.
define i32 @asm_two_fi_large_stack(i32 %a, i32 %b) {
; CHECK-LABEL: asm_two_fi_large_stack:
; CHECK:         {{s[wd]}} a0, {{[0-9]+}}(sp)
; CHECK-NEXT:    lui a0, 1
; CHECK-NEXT:    add a0, sp, a0
; CHECK-NEXT:    {{s[wd]}} a1, {{[0-9]+}}(sp)
; CHECK-NEXT:    lui a1, 1
; CHECK-NEXT:    add a1, sp, a1
; CHECK-NEXT:    #APP
  %x = alloca i32
  %big = alloca [4096 x i8]
  call void asm sideeffect "", "*m,*m,~{x1},~{x5},~{x6},~{x7},~{x8},~{x9},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{x29},~{x30},~{x31}"(ptr elementtype(i32) %x, ptr elementtype(i32) %x)
  %s = add i32 %a, %b
  ret i32 %s
}

; Same asm with a small stack: no scratch register is needed, so the frame
; must not change.
define i32 @asm_two_fi_small_stack(i32 %a, i32 %b) {
; CHECK-LABEL: asm_two_fi_small_stack:
; CHECK-NOT:     lui
; CHECK:         #APP
  %x = alloca i32
  %small = alloca [16 x i8]
  call void asm sideeffect "", "*m,*m,~{x1},~{x5},~{x6},~{x7},~{x8},~{x9},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{x29},~{x30},~{x31}"(ptr elementtype(i32) %x, ptr elementtype(i32) %x)
  %s = add i32 %a, %b
  ret i32 %s
}
