; RUN: llc -mtriple=riscv32 -filetype=obj < %s | llvm-readelf -h - | FileCheck %s

; CHECK: Flags: 0x2, single-float ABI

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"ilp32f"}
