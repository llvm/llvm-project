; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV32I %s
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV64I %s
; RUN: llc -mtriple=riscv32 -mattr=+f -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV32I %s
; RUN: llc -mtriple=riscv64 -mattr=+f -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV64I %s
; RUN: llc -mtriple=riscv32 -mattr=+d -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV32I %s
; RUN: llc -mtriple=riscv64 -mattr=+d -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV64I %s
; RUN: llc -mtriple=riscv32 -mattr=+f -target-abi ilp32f -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV32IF %s
; RUN: llc -mtriple=riscv64 -mattr=+f -target-abi lp64f -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV64IF %s
; RUN: llc -mtriple=riscv32 -mattr=+d -target-abi ilp32d -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV32ID %s
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64d -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV64ID %s


define void @testcase() nounwind {
; RV32I-LABEL: testcase:
; RV32I:      sw s1, {{[0-9]+}}(sp)
; RV32I-NEXT: sw s2, {{[0-9]+}}(sp)
; RV32I-NOT:  fsw fs0, {{[0-9]+}}(sp)
; RV32I-NOT:  fsd fs0, {{[0-9]+}}(sp)
;
; RV64I-LABEL: testcase:
; RV64I:      sd s1, {{[0-9]+}}(sp)
; RV64I-NEXT: sd s2, {{[0-9]+}}(sp)
; RV64I-NOT:  fsw fs0, {{[0-9]+}}(sp)
; RV64I-NOT:  fsd fs0, {{[0-9]+}}(sp)
;
; RV32IF-LABEL: testcase:
; RV32IF:      sw s1, {{[0-9]+}}(sp)
; RV32IF-NEXT: sw s2, {{[0-9]+}}(sp)
; RV32IF-NEXT: fsw fs0, {{[0-9]+}}(sp)
; RV32IF-NEXT: fsw fs1, {{[0-9]+}}(sp)
;
; RV64IF-LABEL: testcase:
; RV64IF:      sd s1, {{[0-9]+}}(sp)
; RV64IF-NEXT: sd s2, {{[0-9]+}}(sp)
; RV64IF-NEXT: fsw fs0, {{[0-9]+}}(sp)
; RV64IF-NEXT: fsw fs1, {{[0-9]+}}(sp)
;
; RV32ID-LABEL: testcase:
; RV32ID:      sw s1, {{[0-9]+}}(sp)
; RV32ID-NEXT: sw s2, {{[0-9]+}}(sp)
; RV32ID-NEXT: fsd fs0, {{[0-9]+}}(sp)
; RV32ID-NEXT: fsd fs1, {{[0-9]+}}(sp)
;
; RV64ID-LABEL: testcase:
; RV64ID:      sd s1, {{[0-9]+}}(sp)
; RV64ID-NEXT: sd s2, {{[0-9]+}}(sp)
; RV64ID-NEXT: fsd fs0, {{[0-9]+}}(sp)
; RV64ID-NEXT: fsd fs1, {{[0-9]+}}(sp)
  tail call void asm sideeffect "", "~{f8},~{f9},~{x9},~{x18}"()
  ret void
}
