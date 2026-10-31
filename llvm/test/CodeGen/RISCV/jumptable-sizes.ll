; RUN: llc -mtriple=riscv32-elf -filetype=obj -code-model=small -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV32I-SMALL
; RUN: llc -mtriple=riscv32-elf -filetype=obj -code-model=medium -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV32I-MEDIUM
; RUN: llc -mtriple=riscv32-elf -filetype=obj -relocation-model=pic -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV32I-PIC
; RUN: llc -mtriple=riscv64-elf -filetype=obj -code-model=small -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV64I-SMALL
; RUN: llc -mtriple=riscv64-elf -filetype=obj -code-model=medium -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV64I-MEDIUM
; RUN: llc -mtriple=riscv64-elf -filetype=obj -relocation-model=pic -verify-machineinstrs < %s \
; RUN:   | llvm-readelf -s - \
; RUN:   | FileCheck %s -check-prefixes=RV64I-PIC


define void @above_threshold(i32 signext %in, ptr %out) nounwind {
; RV32I-SMALL:  24 OBJECT LOCAL DEFAULT [[#]] .LJTI0_0
; RV32I-MEDIUM: 24 OBJECT LOCAL DEFAULT [[#]] .LJTI0_0
; RV32I-PIC:    24 OBJECT LOCAL DEFAULT [[#]] .LJTI0_0
; RV64I-SMALL:  24 OBJECT LOCAL DEFAULT [[#]] .LJTI0_0
; RV64I-MEDIUM: 48 OBJECT LOCAL DEFAULT [[#]] .LJTI0_0
; RV64I-PIC:    24 OBJECT LOCAL DEFAULT [[#]] .LJTI0_0

entry:
  switch i32 %in, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb5
    i32 6, label %bb6
  ]
bb1:
  store i32 4, ptr %out
  br label %exit
bb2:
  store i32 3, ptr %out
  br label %exit
bb3:
  store i32 2, ptr %out
  br label %exit
bb4:
  store i32 1, ptr %out
  br label %exit
bb5:
  store i32 100, ptr %out
  br label %exit
bb6:
  store i32 200, ptr %out
  br label %exit
exit:
  ret void
}
