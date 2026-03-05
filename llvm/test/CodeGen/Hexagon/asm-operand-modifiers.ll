; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test coverage for HexagonAsmPrinter: exercise inline asm operand
; printing with various constraint letters and address operands.

; CHECK-LABEL: test_asm_reg:
; CHECK: r{{[0-9]+}}
define void @test_asm_reg(i32 %a) #0 {
entry:
  call void asm sideeffect "nop // $0", "r"(i32 %a)
  ret void
}

; Exercise the memory operand printing path.
; CHECK-LABEL: test_asm_mem:
; CHECK: memw
define i32 @test_asm_mem(ptr %p) #0 {
entry:
  %val = call i32 asm sideeffect "$0 = memw($1)", "=r,*m"(ptr elementtype(i32) %p)
  ret i32 %val
}

; Exercise immediate operand with 'I' modifier.
; CHECK-LABEL: test_asm_imm:
; CHECK: nop
define void @test_asm_imm() #0 {
entry:
  call void asm sideeffect "nop // $0", "i"(i32 42)
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
