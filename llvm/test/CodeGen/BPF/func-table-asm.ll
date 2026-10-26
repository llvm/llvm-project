; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+func-table < %s | FileCheck %s

; An operand of inline asm is the function, not its index in the table. It
; doesn't have to be static. The same for a function that is marked as used.

; CHECK-LABEL: prog:
; CHECK:       call local
; CHECK:       call global
; CHECK-NOT:   __bpf_func_table

@llvm.compiler.used = appending global [2 x ptr] [ptr @local, ptr @global], section "llvm.metadata"

define internal i64 @local() noinline {
  ret i64 1
}

define i64 @global() noinline {
  ret i64 2
}

define void @prog(ptr %ctx) section "tc" {
  call void asm sideeffect "call $0", "i,~{r0},~{r1},~{r2},~{r3},~{r4},~{r5}"(ptr @local)
  call void asm sideeffect "call $0", "i,~{r0},~{r1},~{r2},~{r3},~{r4},~{r5}"(ptr @global)
  ret void
}
