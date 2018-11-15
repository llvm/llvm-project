; RUN: llc -mtriple=riscv32 -mattr=-relax -filetype=obj < %s \
; RUN:     | llvm-objdump -d -r - | FileCheck %s

; This test demonstrates that .option relax has no effect on codegen
; when emitting an ELF directly.

declare i32 @foo(i32)

define i32 @bar(i32 %a) nounwind {
; CHECK-LABEL: bar:
; CHECK: R_RISCV_CALL
; CHECK-NOT: R_RISCV_RELAX
  tail call void asm sideeffect ".option relax", ""()
  %1 = call i32 @foo(i32 %a)
  ret i32 %1
}
