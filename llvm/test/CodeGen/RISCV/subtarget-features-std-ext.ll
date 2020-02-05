; RUN: llc -mtriple=riscv32 -target-abi ilp32 < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=RV32IF-ILP32 %s
; RUN: llc -mtriple=riscv32 -target-abi ilp32f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=RV32IF-ILP32F %s
; RUN: llc -mtriple=riscv32 -mattr=-f -target-abi ilp32f <%s 2>&1 \
; RUN:   | FileCheck -check-prefix=RV32I-ILP32F-FAILED %s

; RV32I-ILP32F-FAILED: Hard-float 'f' ABI can't be used for a target that doesn't support the F instruction set extension


define float @foo(i32 %a) nounwind #0 {
; RV32IF-ILP32: fcvt.s.w  ft0, a0
; RV32IF-ILP32-NEXT: fmv.x.w a0, ft0
; RV32IF-ILP32F: fcvt.s.w fa0, a0
; RV32IF-ILP32F-NEXT: ret
  %conv = sitofp i32 %a to float
  ret float %conv
}

attributes #0 = { "target-features"="+f"}
