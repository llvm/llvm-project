# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -filetype=obj -o %t/riscv64_reloc_add.o %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj -o %t/riscv32_reloc_add.o %s
# RUN: llvm-jitlink -noexec -check %s %t/riscv64_reloc_add.o
# RUN: llvm-jitlink -noexec -check %s %t/riscv32_reloc_add.o

# jitlink-check: *{8}(named_data) = 0x8
# jitlink-check: *{4}(named_data+8) = 0x8
# jitlink-check: *{2}(named_data+12) = 0x8
# jitlink-check: *{1}(named_data+14) = 0x8

.global main
main:
.L0:
# Referencing named_data symbol to avoid the following .rodata section be skipped.
# This instruction will be expand to two instructions (auipc + ld).
  lw a0, named_data
.L1:

.section ".rodata","",@progbits
.type named_data,@object
named_data:
.dword .L1 - .L0
.word .L1 - .L0
.half .L1 - .L0
.byte .L1 - .L0
.size named_data, 15
