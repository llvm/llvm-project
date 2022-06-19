# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -filetype=obj -o %t/riscv64_reloc_set.o %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj -o %t/riscv32_reloc_set.o %s
# RUN: llvm-jitlink -noexec \
# RUN:  -slab-allocate 100Kb -slab-address 0xfff0f0f0 -slab-page-size 4096 \
# RUN:  -check %s %t/riscv64_reloc_set.o
# RUN: llvm-jitlink -noexec \
# RUN:  -slab-allocate 100Kb -slab-address 0xfff0f0f0 -slab-page-size 4096 \
# RUN:  -check %s %t/riscv32_reloc_set.o

# jitlink-check: *{4}(foo) = foo
# jitlink-check: *{2}(foo+4) = foo[15:0]
# jitlink-check: *{1}(foo+6) = foo[7:0]
# jitlink-check: *{1}(foo+7) = foo[5:0]

.global main
main:
  lw a0, foo

.section ".rodata","",@progbits
.type foo,@object
foo:
  .reloc foo, R_RISCV_SET32, foo
  .reloc foo+4, R_RISCV_SET16, foo
  .reloc foo+6, R_RISCV_SET8, foo
  .reloc foo+7, R_RISCV_SET6, foo
  .word 0
  .half 0
  .byte 0
  .byte 0
  .size foo, 8
