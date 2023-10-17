; RUN: llc -march=nvptx64 < %s | FileCheck %s
; RUN: %if ptxas %{ llc -march=nvptx64 < %s | %ptxas-verify %}

; Test that %c works with immediates
; CHECK-LABEL: test_inlineasm_c_output_template0
; CHECK: //TEST 42
define dso_local i32 @test_inlineasm_c_output_template0() {
  tail call void asm sideeffect "//TEST ${0:c}", "i"(i32 42)
  ret i32 42
}

; Test that %c works with global address
; FIXME: seems this case isn't handled properly by
; SelectionDAG TargetLowering::LowerAsmOperandForConstraint?
; check: test_inlineasm_c_output_template1
; check: //TEST baz
;@baz = internal global i32 0, align 4
;define dso_local i32 @test_inlineasm_c_output_template1() {
;  tail call void asm sideeffect "//TEST ${0:c}", "i"(ptr nonnull @baz)
;  ret i32 42
;}

; Test that %n works with immediates
; CHECK-LABEL: test_inlineasm_c_output_template2
; CHECK: //TEST -42
define dso_local i32 @test_inlineasm_c_output_template2() {
  tail call void asm sideeffect "//TEST ${0:n}", "i"(i32 42)
  ret i32 42
}
