# Test COFF section handling for LoongArch64 UEFI targets

# RUN: llvm-mc -triple=loongarch64-unknown-uefi -filetype=obj -o %t.coff %s
# RUN: llvm-readobj --sections %t.coff | FileCheck %s

.text
.globl _DriverEntry
_DriverEntry:
    addi.d $a0, $zero, 0
    jirl $zero, $ra, 0

.data
.global DriverData
DriverData:
    .dword 0x12345678

.section .rdata
ReadOnlyData:
    .asciz "UEFI Driver"

# CHECK: Sections [
# CHECK:   Name: .text
# CHECK:   Name: .data
# CHECK:   Name: .rdata
# CHECK: ]
