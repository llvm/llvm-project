# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -mattr=+relax -filetype=obj \
# RUN:     -o %t/riscv64_reloc_add.o %s
# RUN: llvm-mc -triple=riscv32 -mattr=+relax -filetype=obj \
# RUN:     -o %t/riscv32_reloc_add.o %s
# RUN: llvm-jitlink -noexec -check %s %t/riscv64_reloc_add.o \
# RUN:     -slab-allocate=1Mb -slab-address=0x1000 -slab-page-size=0x1000
# RUN: llvm-jitlink -noexec -check %s %t/riscv32_reloc_add.o \
# RUN:     -slab-allocate=1Mb -slab-address=0x1000 -slab-page-size=0x1000

# jitlink-check: *{8}(named_data) = 0x8
# jitlink-check: *{4}(named_data+8) = 0x8
# jitlink-check: *{2}(named_data+12) = 0x8
# jitlink-check: *{1}(named_data+14) = 0x8
# jitlink-check: *{1}(named_data+15) = 0x8

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
.reloc named_data+15, R_RISCV_SUB6, .L0
.dword .L1 - .L0
.word .L1 - .L0
.half .L1 - .L0
.byte .L1 - .L0
.byte 0x8
.size named_data, 16
