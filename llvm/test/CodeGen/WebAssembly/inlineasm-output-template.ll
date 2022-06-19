; RUN: llc -mtriple=wasm32 < %s | FileCheck %s

; Skip past the functype directives, which interfere with the CHECK-LABEL
; matches.
;
; Test that %c works with immediates
; CHECK-LABEL: test_inlineasm_c_output_template0:
; CHECK: #TEST 42
define dso_local i32 @test_inlineasm_c_output_template0() {
  tail call void asm sideeffect "#TEST ${0:c}", "i"(i32 42)
  ret i32 42
}

; Test that %c works with global address
; CHECK-LABEL: test_inlineasm_c_output_template2:
; CHECK: #TEST baz
@baz = internal global i32 0, align 4
define dso_local i32 @test_inlineasm_c_output_template2() {
  tail call void asm sideeffect "#TEST ${0:c}", "i"(i32* nonnull @baz)
  ret i32 42
}

; Test that %n works with immediates
; CHECK-LABEL: test_inlineasm_c_output_template1:
; CHECK: #TEST -42
define dso_local i32 @test_inlineasm_c_output_template1() {
  tail call void asm sideeffect "#TEST ${0:n}", "i"(i32 42)
  ret i32 42
}
