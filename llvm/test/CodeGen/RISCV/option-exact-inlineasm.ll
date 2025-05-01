; RUN: llc -mtriple=riscv32 -mattr=+relax,+c %s --filetype=obj -o - \
; RUN:  | llvm-objdump --triple=riscv32 --mattr=+c -M no-aliases -dr - \
; RUN:  | FileCheck %s

define i32 @foo(ptr noundef %f) nounwind {
; CHECK-LABEL: <foo>:
; CHECK:      auipc ra, 0x0
; CHECK-NEXT:     R_RISCV_CALL_PLT undefined
; CHECK-NEXT: jalr ra, 0x0(ra)
; CHECK-NEXT: lw a0, 0x0(a0)
; CHECK-NEXT: c.jr ra

entry:
  %0 = tail call i32 asm sideeffect "
  .option exact
  call undefined@plt
  lw $0, ($1)
  .option noexact", "=^cr,^cr"(ptr %f)
  ret i32 %0
}
