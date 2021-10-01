# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv32 -position-independent -filetype=obj \
# RUN:     -o %t/elf_riscv32_got_plt_reloc.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:     -define-abs external_func=0x1 -define-abs external_data=0x2 \
# RUN:     -check %s %t/elf_riscv32_got_plt_reloc.o

        .text
        .file   "testcase.c"

# Empty main entry point.
        .globl  main
        .p2align  1
        .type   main,@function
main:
        ret

        .size   main, .-main
# Test R_RISCV_GOT_HI20. The low 12 relocation is R_RISCV_PC_REL_LO12. This test case will
# check both the offset to the GOT entry and its content.
# jitlink-check: decode_operand(test_got, 1) = (got_addr(elf_riscv32_got_plt_reloc.o, external_data) - test_got + 0x800)[31:12]
# jitlink-check: decode_operand(test_got+4, 2)[11:0] = (got_addr(elf_riscv32_got_plt_reloc.o, external_data) - test_got)[11:0]
# jitlink-check: *{4}(got_addr(elf_riscv32_got_plt_reloc.o, external_data)) = external_data
        .globl test_got
        .p2align  1
        .type  test_got,@function
test_got:
        auipc a0, %got_pcrel_hi(external_data)
        lw    a0, %pcrel_lo(test_got)(a0)

        .size test_got, .-test_got

# Test R_RISCV_CALL_PLT.
# jitlink-check: decode_operand(test_plt, 1) = (stub_addr(elf_riscv32_got_plt_reloc.o, external_func) - test_plt + 0x800)[31:12]
# jitlink-check: decode_operand(test_plt+4, 2) = (stub_addr(elf_riscv32_got_plt_reloc.o, external_func) - test_plt)[11:0]
# jitlink-check: *{4}(got_addr(elf_riscv32_got_plt_reloc.o, external_func)) = external_func
       .globl   test_plt
       .p2align  1
       .type    test_got,@function
test_plt:
       call external_func@plt

       .size test_plt, .-test_plt
