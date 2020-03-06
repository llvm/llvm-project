; RUN: llc -mtriple=riscv32 -mattr=+c  -filetype=obj < %s\
; RUN: | llvm-objdump -triple=riscv32 -mattr=+c -d -M no-aliases -\
; RUN: | FileCheck -check-prefix=CHECK %s

; This test demonstrates that .option norvc has no effect on codegen when
; emitting an ELF directly.

define i32 @add(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: <add>:
; CHECK:    c.add a0, a1
; CHECK-NEXT:    c.jr ra
  tail call void asm sideeffect ".option norvc", ""()
  %add = add nsw i32 %b, %a
  ret i32 %add
}
