; RUN: llc -mtriple=riscv32 -filetype=obj --relocation-model=pic < %s\
; RUN: | llvm-objdump --triple=riscv32 --mattr=+c -d -M no-aliases -\
; RUN: | FileCheck -check-prefix=CHECK %s

; This test demonstrates that .option nopic has no effect on codegen when
; emitting an ELF directly.

@symbol = global i32 zeroinitializer

define i32 @get_symbol() nounwind {
; CHECK-LABEL: <get_symbol>:
; CHECK: auipc	a0, 0
; CHECK: lw	a0, 0(a0)
; CHECK: lw	a0, 0(a0)
  tail call void asm sideeffect ".option nopic", ""()
  %v = load i32, i32* @symbol
  ret i32 %v
}
