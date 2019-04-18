; RUN: llc -mtriple=ppc32-- < %s | FileCheck %s

; Test that %c works with immediates
; CHECK-LABEL: test_inlineasm_c_output_template0
; CHECK: #TEST 42
define dso_local i32 @test_inlineasm_c_output_template0() {
  tail call void asm sideeffect "#TEST ${0:c}", "i"(i32 42)
  ret i32 42
}

; Test that %n works with immediates
; CHECK-LABEL: test_inlineasm_c_output_template1
; CHECK: #TEST -42
define dso_local i32 @test_inlineasm_c_output_template1() {
  tail call void asm sideeffect "#TEST ${0:n}", "i"(i32 42)
  ret i32 42
}
