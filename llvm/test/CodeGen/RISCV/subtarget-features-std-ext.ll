; RUN: llc -mtriple=riscv32 -target-abi ilp32 < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=RV32IF-ILP32 %s
; RUN: llc -mtriple=riscv32 -target-abi ilp32f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=RV32IF-ILP32F %s

; RV32IF-ILP32F: Hard-float 'f' ABI can't be used for a target that doesn't support the F instruction set extension (ignoring target-abi)

define float @foo(i32 %a) nounwind #0 {
; RV32IF-ILP32: # %bb.0:
; RV32IF-ILP32-NEXT: fcvt.s.w  ft0, a0
  %conv = sitofp i32 %a to float
  ret float %conv
}

attributes #0 = { "target-features"="+f"}
