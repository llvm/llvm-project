# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -filetype=obj -o %t/riscv64_reloc_uleb128.o %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj -o %t/riscv32_reloc_uleb128.o %s
# RUN: llvm-jitlink -noexec -check %s %t/riscv64_reloc_uleb128.o
# RUN: llvm-jitlink -noexec -check %s %t/riscv32_reloc_uleb128.o

# jitlink-check: *{4}(foo+8) = 0x180

.global main
main:
  lw a0, foo

.section ".text","",@progbits
.type foo,@function
foo:
  nop
  nop
  .reloc ., R_RISCV_SET_ULEB128, foo+129
  .reloc ., R_RISCV_SUB_ULEB128, foo+1
  .uleb128 0x80
  .size foo, 8
