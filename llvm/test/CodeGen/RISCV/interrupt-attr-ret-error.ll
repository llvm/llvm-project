; RUN: not llc -mtriple riscv32-unknown-elf -o - %s \
; RUN: 2>&1 | FileCheck %s
; RUN: not llc -mtriple riscv64-unknown-elf -o - %s \
; RUN: 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Functions with the interrupt attribute must have void return type!
define i32 @isr1_user() #0 {
  ret i32 0
}


attributes #0 = { "interrupt"="supervisor" }
