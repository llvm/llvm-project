; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32I
; RUN: llc -mtriple=riscv32 -mattr=+c -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32C
; RUN: llc -filetype=obj -mtriple=riscv32 < %s -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefixes=SEC,SEC-I
; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=+c < %s -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefixes=SEC,SEC-C

; SEC:   Name   Type     Address  Off      Size     ES Flg Lk Inf Al
; SEC-I: .text  PROGBITS 00000000 [[#%x,]] [[#%x,]] 00  AX  0   0  4
; SEC-C: .text  PROGBITS 00000000 [[#%x,]] [[#%x,]] 00  AX  0   0  2

define void @foo() {
;RV32I: .p2align 2
;RV32I: foo:
;RV32C: .p2align 1
;RV32C: foo:
entry:
  ret void
}
