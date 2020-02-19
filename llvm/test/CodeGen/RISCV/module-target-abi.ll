; RUN: llc -mtriple=riscv32 < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -mtriple=riscv32 -target-abi ilp32 < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=RV32IF-ILP32 %s
; RUN: not --crash llc -mtriple=riscv32 -target-abi ilp32f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=RV32IF-ILP32F %s
; RUN: llc -mtriple=riscv32 -filetype=obj < %s | llvm-readelf -h - | FileCheck -check-prefixes=FLAGS %s

; RV32IF-ILP32F: -target-abi option != target-abi module flag

; FLAGS: Flags: 0x0

define float @foo(i32 %a) nounwind #0 {
; DEFAULT: # %bb.0:
; DEFAULT: fmv.x.w a0, ft0
; RV32IF-ILP32: # %bb.0:
; RV32IF-ILP32: fmv.x.w a0, ft0
  %conv = sitofp i32 %a to float
  ret float %conv
}

attributes #0 = { "target-features"="+f"}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"ilp32"}
