;; Generate ELF attributes from llc.

; RUN: llc -mtriple=riscv32 -mattr=+xsfvfwmaccqqq %s -o - | FileCheck --check-prefix=RV32XSFVFWMACCQQQ %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfvfwmaccqqq %s -o - | FileCheck --check-prefix=RV64XSFVFWMACCQQQ %s

; RUN: llc -mtriple=riscv32 -mattr=+xsfmm128t %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM128T %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm16t %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM16T %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32a8i %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32A8I %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32a8f %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32A8F %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32a16f %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32A16F %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32a32f %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32A32F %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32t %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32T %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm64a64f %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM64A64F %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm64t %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM64T %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmmbase %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMMBASE %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm128t %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM128T %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm16t %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM16T %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32a8i %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32A8I %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32a8f %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32A8F %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32a16f %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32A16F %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32a32f %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32A32F %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32t %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32T %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm64a64f %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM64A64F %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm64t %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM64T %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmmbase %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMMBASE %s

; CHECK: .attribute 4, 16

; RV32XSFVFWMACCQQQ: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvfbfmin1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfvfwmaccqqq1p0"
; RV64XSFVFWMACCQQQ: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvfbfmin1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfvfwmaccqqq1p0"

; RV32XSFMM128T: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0_xsfmm128t0p6_xsfmmbase0p6"
; RV32XSFMM16T: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_zvl64b1p0_xsfmm16t0p6_xsfmmbase0p6" 
; RV32XSFMM32A8I: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xsfmm32a8i0p6_xsfmmbase0p6"
; RV32XSFMM32A8F: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfmm32a8f0p6_xsfmmbase0p6"
; RV32XSFMM32A16F: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfmm32a16f0p6_xsfmmbase0p6"
; RV32XSFMM32A32F: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfmm32a32f0p6_xsfmmbase0p6"
; RV32XSFMM32T: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32t0p6_xsfmmbase0p6"
; RV32XSFMM64A64F: .attribute 5, "rv32i2p1_f2p2_d2p2_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0_xsfmm64a64f0p6_xsfmmbase0p6"
; RV32XSFMM64T: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0_xsfmm64t0p6_xsfmmbase0p6"
; RV32XSFMMBASE: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xsfmmbase0p6"
; RV64XSFMM128T: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0_xsfmm128t0p6_xsfmmbase0p6"
; RV64XSFMM16T: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_zvl64b1p0_xsfmm16t0p6_xsfmmbase0p6"
; RV64XSFMM32A8I: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xsfmm32a8i0p6_xsfmmbase0p6"
; RV64XSFMM32A8F: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfmm32a8f0p6_xsfmmbase0p6"
; RV64XSFMM32A16F: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfmm32a16f0p6_xsfmmbase0p6"
; RV64XSFMM32A32F: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfmm32a32f0p6_xsfmmbase0p6"
; RV64XSFMM32T: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32t0p6_xsfmmbase0p6"
; RV64XSFMM64A64F: .attribute 5, "rv64i2p1_f2p2_d2p2_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0_xsfmm64a64f0p6_xsfmmbase0p6"
; RV64XSFMM64T: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0_xsfmm64t0p6_xsfmmbase0p6"
; RV64XSFMMBASE: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xsfmmbase0p6"

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}
