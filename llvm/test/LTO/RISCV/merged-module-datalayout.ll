; Check that legacy LTO preserves the input module's DataLayout
; instead of overwriting it with one recomputed from the
; TargetMachine. The ilp32e ABI is carried by the target-abi module
; flag, so the TargetMachine's option-derived DataLayout would use the
; default -S128 stack alignment rather than the -S32 the module
; actually requires.

; RUN: llvm-as %s -o %t.o
; RUN: llvm-lto -save-merged-module -o %t.elf %t.o
; RUN: llvm-dis %t.elf.merged.bc -o - | FileCheck %s

; CHECK: target datalayout = "e-m:e-p:32:32-i64:64-n32-S32"

target datalayout = "e-m:e-p:32:32-i64:64-n32-S32"
target triple = "riscv32-unknown-unknown-elf"

define dso_local i32 @_start() #0 {
entry:
  ret i32 0
}

attributes #0 = { "target-cpu"="generic-rv32" "target-features"="+32bit,+e" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"target-abi", !"ilp32e"}
