## Test that non-relaxable edges have their offset adjusted by relaxation

# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o %t.rv32
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x0 -slab-page-size 4096 \
# RUN:     -check %s %t.rv32

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.rv64
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x0 -slab-page-size 4096 \
# RUN:     -check %s %t.rv64

    .globl main,nonrelaxable,nonrelaxable_target
    .size nonrelaxable, 4
    .size nonrelaxable_target, 4
main:
    call f
nonrelaxable:
    ## Non-relaxable R_RISCV_BRANCH edge after a relaxable R_RISCV_CALL edge.
    ## Even though this edge isn't relaxable, its offset should still be
    ## adjusted.
    beq zero, zero, nonrelaxable_target
nonrelaxable_target:
    ret
    .size main, .-main

    .globl f
f:
    ret
    .size f, .-f

# jitlink-check: decode_operand(nonrelaxable, 2) = (nonrelaxable_target - nonrelaxable)

