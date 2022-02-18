# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -filetype=obj \
# RUN:     -o %t/elf_riscv64_jal.o %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj \
# RUN:     -o %t/elf_riscv32_jal.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1ff00000 -slab-page-size 4096 \
# RUN:     -abs external_func=0x1fe000fe \
# RUN:     -check %s %t/elf_riscv64_jal.o
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1ff00000 -slab-page-size 4096 \
# RUN:     -abs external_func=0x1fe000fe \
# RUN:     -check %s %t/elf_riscv32_jal.o
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

# Test R_RISCV_JAL

# jitlink-check: decode_operand(test_jal, 1)[31:12] = (external_func - test_jal)[31:12]
  .globl  test_jal
  .p2align  1
  .type  test_jal,@function
test_jal:
  jal	x0, external_func

  .size test_jal, .-test_jal
