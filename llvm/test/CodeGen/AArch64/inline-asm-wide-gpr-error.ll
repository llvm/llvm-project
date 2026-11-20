; RUN: not llc -mtriple=aarch64-linux-gnu < %s 2>&1 | FileCheck %s

; %H names the second register of an operand that takes two, such as an i128,
; so it's an error on an operand that takes one, or more than two.

; CHECK: error: invalid operand in inline asm: '// ${0:H}'
define void @test_one_register(i64 %x) {
entry:
  call void asm sideeffect "// ${0:H}", "r"(i64 %x)
  ret void
}

; CHECK: error: invalid operand in inline asm: '// ${0:H} four'
define void @test_four_registers(i256 %x) {
entry:
  call void asm sideeffect "// ${0:H} four", "r"(i256 %x)
  ret void
}
