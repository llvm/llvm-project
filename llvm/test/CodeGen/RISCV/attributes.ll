;; Generate ELF attributes from llc.

; RUN: llc -mtriple=riscv32 -mattr=+m %s -o - | FileCheck --check-prefix=RV32M %s
; RUN: llc -mtriple=riscv32 -mattr=+a %s -o - | FileCheck --check-prefix=RV32A %s
; RUN: llc -mtriple=riscv32 -mattr=+f %s -o - | FileCheck --check-prefix=RV32F %s
; RUN: llc -mtriple=riscv32 -mattr=+d %s -o - | FileCheck --check-prefix=RV32D %s
; RUN: llc -mtriple=riscv32 -mattr=+c %s -o - | FileCheck --check-prefix=RV32C %s
; RUN: llc -mtriple=riscv64 -mattr=+m %s -o - | FileCheck --check-prefix=RV64M %s
; RUN: llc -mtriple=riscv64 -mattr=+a %s -o - | FileCheck --check-prefix=RV64A %s
; RUN: llc -mtriple=riscv64 -mattr=+f %s -o - | FileCheck --check-prefix=RV64F %s
; RUN: llc -mtriple=riscv64 -mattr=+d %s -o - | FileCheck --check-prefix=RV64D %s
; RUN: llc -mtriple=riscv64 -mattr=+c %s -o - | FileCheck --check-prefix=RV64C %s

; RV32M: .attribute 5, "rv32i2p0_m2p0"
; RV32A: .attribute 5, "rv32i2p0_a2p0"
; RV32F: .attribute 5, "rv32i2p0_f2p0"
; RV32D: .attribute 5, "rv32i2p0_f2p0_d2p0"
; RV32C: .attribute 5, "rv32i2p0_c2p0"
; RV64M: .attribute 5, "rv64i2p0_m2p0"
; RV64A: .attribute 5, "rv64i2p0_a2p0"
; RV64F: .attribute 5, "rv64i2p0_f2p0"
; RV64D: .attribute 5, "rv64i2p0_f2p0_d2p0"
; RV64C: .attribute 5, "rv64i2p0_c2p0"

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}
