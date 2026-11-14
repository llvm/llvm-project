; RUN: llc -mtriple=riscv32-elf -mattr=+d -filetype=obj -code-model=small -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV32I-SMALL
; RUN: llc -mtriple=riscv32-elf -mattr=+d -filetype=obj -code-model=medium -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV32I-MEDIUM
; RUN: llc -mtriple=riscv32-elf -mattr=+d -filetype=obj -relocation-model=pic -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV32I-PIC
; RUN: llc -mtriple=riscv64-elf -mattr=+d -filetype=obj -code-model=small -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV64I-SMALL
; RUN: llc -mtriple=riscv64-elf -mattr=+d -filetype=obj -code-model=medium -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV64I-MEDIUM
; RUN: llc -mtriple=riscv64-elf -mattr=+d -filetype=obj -relocation-model=pic -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV64I-PIC

define double @add_double_constant(double %a) nounwind {
; RV32I-SMALL:  8 OBJECT LOCAL DEFAULT [[#]] .LCPI0_0
; RV32I-MEDIUM: 8 OBJECT LOCAL DEFAULT [[#]] .LCPI0_0
; RV32I-PIC:    8 OBJECT LOCAL DEFAULT [[#]] .LCPI0_0
; RV64I-SMALL:  8 OBJECT LOCAL DEFAULT [[#]] .LCPI0_0
; RV64I-MEDIUM: 8 OBJECT LOCAL DEFAULT [[#]] .LCPI0_0
; RV64I-PIC:    8 OBJECT LOCAL DEFAULT [[#]] .LCPI0_0

  %1 = fadd double %a, 0x400921FB54442D18
  ret double %1
}
