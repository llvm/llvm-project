; RUN: not --crash llc -mtriple riscv32-unknown-elf -o - %s \
; RUN: 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple riscv64-unknown-elf -o - %s \
; RUN: 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Functions with the interrupt attribute cannot have arguments!
define i32 @isr_user(i8 %n) #0 {
  ret i32 0
}

attributes #0 = { "interrupt"="supervisor" }
