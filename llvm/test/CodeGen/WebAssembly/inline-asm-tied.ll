; RUN: llc < %s -mtriple=wasm32-unknown-unknown -O0 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=wasm64-unknown-unknown -O0 -verify-machineinstrs | FileCheck %s

; Test that tied operands in inline assembly are lowered correctly at -O0.
; Because WebAssembly does not run RegAllocFast, TwoAddressInstructionPass must
; run to insert the copy for the tied operand.

; CHECK-LABEL: tied_operands:
; CHECK:      local.get 0
; CHECK-NEXT: local.set 1
; CHECK:      local.get 1
; CHECK-NEXT: return
define i32 @tied_operands(i32 %var) {
entry:
  %ret = call i32 asm "", "=r,0"(i32 %var)
  ret i32 %ret
}
