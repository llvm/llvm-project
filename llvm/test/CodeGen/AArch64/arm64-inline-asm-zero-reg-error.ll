; RUN: not llc < %s -mtriple=arm64-eabi 2>&1 | FileCheck %s


; The 'z' constraint allocates either xzr or wzr, but obviously an input of 1 is
; incompatible.
define void @test_bad_zero_reg() {
  tail call void asm sideeffect "USE($0)", "z"(i32 1) nounwind
; CHECK: error: invalid operand for inline asm constraint 'z'

  ret void
}
