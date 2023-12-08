; RUN: llc < %s -march=bpfel -mattr=+alu32 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=bpfeb -mattr=+alu32 -verify-machineinstrs | FileCheck %s

; Test that %w works as input constraint
; CHECK-LABEL: test_inlineasm_w_input_constraint
define dso_local i32 @test_inlineasm_w_input_constraint() {
  tail call void asm sideeffect "w0 = $0", "w"(i32 42)
; CHECK: w0 = w1
  ret i32 42
}

; Test that %w works as output constraint
; CHECK-LABEL: test_inlineasm_w_output_constraint
define dso_local i32 @test_inlineasm_w_output_constraint() {
  %1 = tail call i32 asm sideeffect "$0 = $1", "=w,i"(i32 42)
; CHECK: w0 = 42
  ret i32 %1
}
