# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -filetype=obj -o %t/riscv64_pc_relative.o %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj -o %t/riscv32_pc_relative.o %s
# RUN: llvm-jitlink -noexec -check %s %t/riscv64_pc_relative.o
# RUN: llvm-jitlink -noexec -check %s %t/riscv32_pc_relative.o

# jitlink-check: *{4}(foo) = 0x4

.global main
main:
  lw a0, foo

.section ".text","",@progbits
.type foo,@function
foo:
  nop
  nop
  .reloc foo, R_RISCV_32_PCREL, foo+4
  .size foo, 8
