# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -filetype=obj \
# RUN:     -o %t/elf_riscv64_branch.o %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj \
# RUN:     -o %t/elf_riscv32_branch.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:     -abs external_func=0xfe \
# RUN:     -check %s %t/elf_riscv64_branch.o
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:     -abs external_func=0xfe \
# RUN:     -check %s %t/elf_riscv32_branch.o
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

# Test R_RISCV_BRANCH

# jitlink-check: decode_operand(test_branch, 2)[11:0] = (external_func - test_branch)[11:0]
  .globl  test_branch
  .p2align  1
  .type  test_branch,@function
test_branch:
  bge	a0, a1, external_func

  .size test_branch, .-test_branch
