# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -filetype=obj \
# RUN:     -o %t/elf_riscv64_non_pc_indirect_reloc.o %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj \
# RUN:     -o %t/elf_riscv32_non_pc_indirect_reloc.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1ff00000 -slab-page-size 4096 \
# RUN:     -abs external_data=0x1ff11115 \
# RUN:     -check %s %t/elf_riscv64_non_pc_indirect_reloc.o
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1ff00000 -slab-page-size 4096 \
# RUN:     -abs external_data=0x1ff11115 \
# RUN:     -check %s %t/elf_riscv32_non_pc_indirect_reloc.o
#

        .text
        .file   "testcase.c"

# Empty main entry point.
        .globl  main
        .p2align  1
        .type   main,@function
main:
        ret

        .size   main, .-main

# Test R_RISCV_HI20 and R_RISCV_LO12

# jitlink-check: decode_operand(test_abs_rel, 1) = (external_data + 0x800)[31:12]
# jitlink-check: decode_operand(test_abs_rel+4, 2)[11:0] = (external_data)[11:0]
# jitlink-check: decode_operand(test_abs_rel+8, 2)[11:0] = (external_data)[11:0]
  .globl  test_abs_rel
  .p2align  1
  .type  test_abs_rel,@function
test_abs_rel:
  lui  a1, %hi(external_data)
  lw  a0, %lo(external_data)(a1)
  sw  a0, %lo(external_data)(a1)

  .size test_abs_rel, .-test_abs_rel
