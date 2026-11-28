; RUN: not llc -mtriple=riscv64 %s -o - 2>&1 | FileCheck %s

; An interrupt handler must preserve all registers; preserve_none can't be
; combined with it.

; CHECK: LLVM ERROR: Functions with the interrupt attribute cannot use the preserve_none calling convention!

define preserve_nonecc void @handler() "interrupt"="machine" {
  ret void
}
