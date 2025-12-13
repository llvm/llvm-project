; RUN: not llc -mtriple riscv32-unknown-elf -mattr=-smrnmi -o - %s 2>&1 \
; RUN:   | FileCheck %s
; RUN: not llc -mtriple riscv64-unknown-elf -mattr=-smrnmi -o - %s 2>&1 \
; RUN:   | FileCheck %s

; CHECK: LLVM ERROR: 'rnmi' interrupt kind requires Srnmi extension
define void @test_rnmi() "interrupt"="rnmi" {
  ret void
}
