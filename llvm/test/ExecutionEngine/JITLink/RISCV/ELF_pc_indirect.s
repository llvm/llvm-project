# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -position-independent -filetype=obj \
# RUN:     -o %t/elf_riscv64_sm_pic_reloc.o %s
# RUN: llvm-mc -triple=riscv32 -position-independent -filetype=obj \
# RUN:     -o %t/elf_riscv32_sm_pic_reloc.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1ff00000 -slab-page-size 4096 \
# RUN:     -abs external_data=0x2 \
# RUN:     -check %s %t/elf_riscv64_sm_pic_reloc.o
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1ff00000 -slab-page-size 4096 \
# RUN:     -abs external_data=0x2 \
# RUN:     -check %s %t/elf_riscv32_sm_pic_reloc.o
#
# Test ELF small/PIC relocations

        .text
        .file   "testcase.c"

# Empty main entry point.
        .globl  main
        .p2align  1
        .type   main,@function
main:
        ret

        .size   main, .-main

# Test R_RISCV_PCREL_HI20 and R_RISCV_PCREL_LO
# jitlink-check: decode_operand(test_pcrel32, 1) = ((external_data - test_pcrel32) + 0x800)[31:12]
# jitlink-check: decode_operand(test_pcrel32+4, 2)[11:0] = (external_data - test_pcrel32)[11:0]
        .globl  test_pcrel32
        .p2align  1
        .type   test_pcrel32,@function
test_pcrel32:
        auipc a0, %pcrel_hi(external_data)
        lw  a0, %pcrel_lo(test_pcrel32)(a0)

        .size   test_pcrel32, .-test_pcrel32

# Test R_RISCV_PCREL_HI20 and R_RISCV_PCREL_LO12_S
# jitlink-check: decode_operand(test_pcrel32_s, 1) = ((external_data - test_pcrel32_s) + 0x800)[31:12]
# jitlink-check: decode_operand(test_pcrel32_s+4, 2)[11:0] = (external_data - test_pcrel32_s)[11:0]
        .globl  test_pcrel32_s
        .p2align  1
        .type   test_pcrel32_s,@function
test_pcrel32_s:
        auipc a0, %pcrel_hi(external_data)
        sw  a0, %pcrel_lo(test_pcrel32_s)(a0)

        .size   test_pcrel32_s, .-test_pcrel32_s

# Test R_RISCV_CALL
# jitlink-check: decode_operand(test_call, 1) = ((internal_func - test_call) + 0x800)[31:12]
# jitlink-check: decode_operand(test_call+4, 2)[11:0] = (internal_func - test_call)[11:0]
        .globl test_call
        .p2align  1
        .type  test_call,@function
test_call:
        .reloc ., R_RISCV_CALL, internal_func
        auipc ra, 0
        jalr ra
        ret
        .size test_call, .-test_call

# Test R_RISCV_CALL_PLT
# jitlink-check: decode_operand(test_call_plt, 1) = ((internal_func - test_call_plt) + 0x800)[31:12]
# jitlink-check: decode_operand(test_call_plt+4, 2)[11:0] = (internal_func - test_call_plt)[11:0]
        .globl test_call_plt
        .p2align  1
        .type  test_call_plt,@function
test_call_plt:
        .reloc ., R_RISCV_CALL_PLT, internal_func
        auipc ra, 0
        jalr ra
        ret
        .size test_call_plt, .-test_call_plt

       .globl   internal_func
       .p2align  1
       .type    internal_func,@function
internal_func:
       ret
       .size internal_func, .-internal_func
