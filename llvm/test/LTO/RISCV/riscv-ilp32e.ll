; Check that we don't crash on DataLayout incompatibility issue.
; RUN: llvm-as %s -o %t.o
; RUN: llvm-lto2 run -r %t.o,_start %t.o -o %t.elf
; RUN: llvm-readobj -h %t.elf.0 | FileCheck %s --check-prefixes=CHECK
; CHECK:  Machine: EM_RISCV (0xF3)
; CHECK:  EF_RISCV_RVE (0x8)


target datalayout = "e-m:e-p:32:32-i64:64-n32-S32"
target triple = "riscv32-unknown-unknown-elf"

define dso_local i32 @_start() #0 {
entry:
  ret i32 0
}

attributes #0 = { "target-cpu"="generic-rv32" "target-features"="+32bit,+e" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"target-abi", !"ilp32e"}
