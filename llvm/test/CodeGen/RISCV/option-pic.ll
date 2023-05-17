; RUN: llc -mtriple=riscv32 -filetype=obj < %s\
; RUN: | llvm-objdump --triple=riscv32 --mattr=+c -d -M no-aliases -\
; RUN: | FileCheck -check-prefix=CHECK %s

; This test demonstrates that .option pic has no effect on codegen when
; emitting an ELF directly.

@symbol = global i32 zeroinitializer

define i32 @get_symbol() nounwind {
; CHECK-LABEL: <get_symbol>:
; CHECK: lui	a0, 0
; CHECK: lw	a0, 0(a0)
  tail call void asm sideeffect ".option pic", ""()
  %v = load i32, ptr @symbol
  ret i32 %v
}
