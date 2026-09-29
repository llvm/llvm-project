; RUN: llc -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=RV32
; RUN: llc -mtriple=riscv64 %s -o - | FileCheck %s --check-prefix=RV64

; Test generation of ELF attribute from module metadata

; RV32: .attribute 5, "rv32i2p1_m2p0_zmmul1p0_zba1p0"
; RV64: .attribute 5, "rv64i2p1_m2p0_zmmul1p0_zba1p0"

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}

!llvm.module.flags = !{!0}

!0 = !{i32 6, !"riscv-isa", !1}
!1 = !{!"rv64i2p1_m2p0", !"rv64i2p1_zba1p0"}
