# RUN: llvm-mc --triple=riscv32-unknown-none-elf %s -filetype=obj -o - \
# RUN: | llvm-objdump -dr - \
# RUN: | FileCheck %s
# RUN: llvm-mc --triple=riscv64-unknown-none-elf %s -filetype=obj -o - \
# RUN: | llvm-objdump -dr - \
# RUN: | FileCheck %s


  # CHECK: 00000013 nop
  nop

  # CHECK-NEXT: 55 55 55 55 .word 0x55555555
  .word 0x55555555

  # CHECK-NEXT: 00 00 00 00 .word 0x00000000
  # CHECK-NEXT: R_RISCV_32 foo
  .word foo

  # CHECK-NEXT: 00000013 nop
  nop
