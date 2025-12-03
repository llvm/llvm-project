; Check that we error out if tail is not possible but call is marked as mustail.

; RUN: not --crash llc -mtriple riscv32-unknown-linux-gnu -o - %s \
; RUN: 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple riscv32-unknown-elf -o - %s \
; RUN: 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple riscv64-unknown-linux-gnu -o - %s \
; RUN: 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple riscv64-unknown-elf -o - %s \
; RUN: 2>&1 | FileCheck %s

declare void @callee_musttail()

define void @caller_musttail() #0 {
; CHECK: LLVM ERROR: failed to perform tail call elimination on a call site marked musttail
entry:
  musttail call void @callee_musttail()
  ret void
}

attributes #0 = { "interrupt"="machine" }
