; The "target-abi" module flag selects the ABI used for codegen, and must also
; drive the ABI recorded in the ELF header e_flags.
; RUN: llc --mtriple=loongarch64 --mattr=+d --filetype=obj < %s -o %t.o
; RUN: llvm-readobj -h %t.o | FileCheck %s

; CHECK: EF_LOONGARCH_ABI_SOFT_FLOAT

define void @nothing() nounwind {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"lp64s"}
