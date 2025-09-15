;; Generate ELF attributes from llc.

; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqccmp %s -o - | FileCheck --check-prefix=RV32XQCCMP %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcia %s -o - | FileCheck --check-prefix=RV32XQCIA %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqciac %s -o - | FileCheck --check-prefix=RV32XQCIAC %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcibi %s -o - | FileCheck --check-prefix=RV32XQCIBI %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcibm %s -o - | FileCheck --check-prefix=RV32XQCIBM %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcicli %s -o - | FileCheck --check-prefix=RV32XQCICLI %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcicm %s -o - | FileCheck --check-prefix=RV32XQCICM %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcics %s -o - | FileCheck --check-prefix=RV32XQCICS %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcicsr %s -o - | FileCheck --check-prefix=RV32XQCICSR %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqciint %s -o - | FileCheck --check-prefix=RV32XQCIINT %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqciio %s -o - | FileCheck --check-prefix=RV32XQCIIO %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcilb %s -o - | FileCheck --check-prefix=RV32XQCILB %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcili %s -o - | FileCheck --check-prefix=RV32XQCILI %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcilia %s -o - | FileCheck --check-prefix=RV32XQCILIA %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcilo %s -o - | FileCheck --check-prefix=RV32XQCILO %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcilsm %s -o - | FileCheck --check-prefix=RV32XQCILSM %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcisim %s -o - | FileCheck --check-prefix=RV32XQCISIM %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcisls %s -o - | FileCheck --check-prefix=RV32XQCISLS %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xqcisync %s -o - | FileCheck --check-prefix=RV32XQCISYNC %s

; RUN: llc -mtriple=riscv64 -mattr=+experimental-xqccmp %s -o - | FileCheck --check-prefix=RV64XQCCMP %s

; RV32XQCCMP: .attribute 5, "rv32i2p1_zca1p0_xqccmp0p3"
; RV32XQCIA: .attribute 5, "rv32i2p1_xqcia0p7"
; RV32XQCIAC: .attribute 5, "rv32i2p1_zca1p0_xqciac0p3"
; RV32XQCIBI: .attribute 5, "rv32i2p1_zca1p0_xqcibi0p2"
; RV32XQCIBM: .attribute 5, "rv32i2p1_zca1p0_xqcibm0p8"
; RV32XQCICLI: .attribute 5, "rv32i2p1_xqcicli0p3"
; RV32XQCICM: .attribute 5, "rv32i2p1_zca1p0_xqcicm0p2"
; RV32XQCICS: .attribute 5, "rv32i2p1_xqcics0p2"
; RV32XQCICSR: .attribute 5, "rv32i2p1_xqcicsr0p4"
; RV32XQCIINT: .attribute 5, "rv32i2p1_zca1p0_xqciint0p10"
; RV32XQCIIO: .attribute 5, "rv32i2p1_xqciio0p1"
; RV32XQCILB: .attribute 5, "rv32i2p1_zca1p0_xqcilb0p2"
; RV32XQCILI: .attribute 5, "rv32i2p1_zca1p0_xqcili0p2"
; RV32XQCILIA: .attribute 5, "rv32i2p1_zca1p0_xqcilia0p2"
; RV32XQCILO: .attribute 5, "rv32i2p1_zca1p0_xqcilo0p3"
; RV32XQCILSM: .attribute 5, "rv32i2p1_xqcilsm0p6"
; RV32XQCISIM: attribute 5, "rv32i2p1_zca1p0_xqcisim0p2"
; RV32XQCISLS: .attribute 5, "rv32i2p1_xqcisls0p2"
; RV32XQCISYNC: attribute 5, "rv32i2p1_zca1p0_xqcisync0p3"

; RV64XQCCMP: .attribute 5, "rv64i2p1_zca1p0_xqccmp0p3"

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}
