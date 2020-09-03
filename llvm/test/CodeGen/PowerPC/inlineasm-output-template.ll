; RUN: llc -mtriple=ppc32 < %s | FileCheck %s
; RUN: llc -mtriple=ppc64 < %s | FileCheck %s --check-prefix=PPC64

; Test that %c works with immediates
; CHECK-LABEL: test_inlineasm_c_output_template0
; CHECK: #TEST 42
define dso_local i32 @test_inlineasm_c_output_template0() {
  tail call void asm sideeffect "#TEST ${0:c}", "i"(i32 42)
  ret i32 42
}

; Test that %c works with global address
; CHECK-LABEL: test_inlineasm_c_output_template1:
; CHECK: #TEST baz
@baz = internal global i32 0, align 4
define dso_local i32 @test_inlineasm_c_output_template1() {
  tail call void asm sideeffect "#TEST ${0:c}", "i"(i32* nonnull @baz)
  ret i32 43
}

; Test that %n works with immediates
; CHECK-LABEL: test_inlineasm_c_output_template2
; CHECK: #TEST -42
define dso_local i32 @test_inlineasm_c_output_template2() {
  tail call void asm sideeffect "#TEST ${0:n}", "i"(i32 42)
  ret i32 42
}

; Test that the machine specific %L works with memory operands.
; CHECK-LABEL: test_inlineasm_L_output_template
; CHECK: # 4(5)
; PPC64-LABEL: test_inlineasm_L_output_template
; PPC64: # 8(4)
define dso_local void @test_inlineasm_L_output_template(i64 %0, i64* %1) {
  tail call void asm sideeffect "# ${0:L}", "*m"(i64* %1)
  ret void
}
