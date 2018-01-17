# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump -mattr=+c -d - | FileCheck -check-prefix=CHECK-INST %s

# alpha and main are 8 byte alignment
# but the alpha function's size is 6
# So assembler will insert a c.nop to make sure 8 byte alignment.

        .text
       .p2align        3
       .type   alpha,@function
alpha:
# BB#0:
       addi    sp, sp, -16
       c.lw    a0, 0(a0)
# CHECK-INST: c.nop
.Lfunc_end0:
       .size   alpha, .Lfunc_end0-alpha
                                        # -- End function
       .globl  main
       .p2align        3
       .type   main,@function
main:                                   # @main
# BB#0:
.Lfunc_end1:
       .size   main, .Lfunc_end1-main
                                        # -- End function
