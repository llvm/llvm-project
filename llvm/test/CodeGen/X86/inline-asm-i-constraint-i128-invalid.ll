; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -filetype=null < %s 2>&1 | FileCheck %s

; PR173841
define i32 @test_constant_does_not_fit_into_64_bits() {
; CHECK: error: <unknown>:0:0: in function test_constant_does_not_fit_into_64_bits i32 (): unsupported size for integer operand
entry:
  tail call void asm sideeffect "movq $0, %rax", "i,~{dirflag},~{fpsr},~{flags}"(i128 18446744073709551616)
  ret i32 0
}

; PR173841
define i32 @test_negative_constant_does_not_fit_into_64_bits() {
; CHECK: error: <unknown>:0:0: in function test_negative_constant_does_not_fit_into_64_bits i32 (): unsupported size for integer operand
entry:
  tail call void asm sideeffect "movq $0, %rax", "i,~{dirflag},~{fpsr},~{flags}"(i128 -9223372036854775809)
  ret i32 0
}
