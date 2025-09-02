;; Generate ELF attributes from llc.

; RUN: llc -mtriple=riscv32 -mattr=+xandesperf %s -o - | FileCheck --check-prefix=RV32XANDESPERF %s
; RUN: llc -mtriple=riscv32 -mattr=+xandesbfhcvt %s -o - | FileCheck --check-prefix=RV32XANDESBFHCVT %s
; RUN: llc -mtriple=riscv32 -mattr=+xandesvbfhcvt %s -o - | FileCheck --check-prefix=RV32XANDESVBFHCVT %s
; RUN: llc -mtriple=riscv32 -mattr=+xandesvsintload %s -o - | FileCheck --check-prefix=RV32XANDESVSINTLOAD %s
; RUN: llc -mtriple=riscv32 -mattr=+xandesvdot %s -o - | FileCheck --check-prefix=RV32XANDESVDOT %s
; RUN: llc -mtriple=riscv32 -mattr=+xandesvpackfph %s -o - | FileCheck --check-prefix=RV32XANDESVPACKFPH %s

; RUN: llc -mtriple=riscv64 -mattr=+xandesperf %s -o - | FileCheck --check-prefix=RV64XANDESPERF %s
; RUN: llc -mtriple=riscv64 -mattr=+xandesbfhcvt %s -o - | FileCheck --check-prefix=RV64XANDESBFHCVT %s
; RUN: llc -mtriple=riscv64 -mattr=+xandesvbfhcvt %s -o - | FileCheck --check-prefix=RV64XANDESVBFHCVT %s
; RUN: llc -mtriple=riscv64 -mattr=+xandesvsintload %s -o - | FileCheck --check-prefix=RV64XANDESVSINTLOAD %s
; RUN: llc -mtriple=riscv64 -mattr=+xandesvdot %s -o - | FileCheck --check-prefix=RV64XANDESVDOT %s
; RUN: llc -mtriple=riscv64 -mattr=+xandesvpackfph %s -o - | FileCheck --check-prefix=RV64XANDESVPACKFPH %s

; RV32XANDESPERF: .attribute 5, "rv32i2p1_xandesperf5p0"
; RV32XANDESBFHCVT: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_xandesbfhcvt5p0"
; RV32XANDESVBFHCVT: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xandesvbfhcvt5p0"
; RV32XANDESVSINTLOAD: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xandesvsintload5p0"
; RV32XANDESVDOT: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xandesvdot5p0"
; RV32XANDESVPACKFPH: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_xandesvpackfph5p0"

; RV64XANDESPERF: .attribute 5, "rv64i2p1_xandesperf5p0"
; RV64XANDESBFHCVT: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_xandesbfhcvt5p0"
; RV64XANDESVBFHCVT: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xandesvbfhcvt5p0"
; RV64XANDESVSINTLOAD: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xandesvsintload5p0"
; RV64XANDESVDOT: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xandesvdot5p0"
; RV64XANDESVPACKFPH: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_xandesvpackfph5p0"

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}
