## Test that we can handle R_RISCV_ALIGN.

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+c %s -o %t.rv32
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x0 -slab-page-size 4096 \
# RUN:     -check %s %t.rv32

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+c %s -o %t.rv64
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x0 -slab-page-size 4096 \
# RUN:     -check %s %t.rv64

    .globl main,align2,align4,align8,align16,align32
    .type  main,@function
main:
    jump f, t0
    .balign 2
align2:
    jump f, t0
    .size align2,.-align2
    .balign 4
align4:
    jump f, t0
    .size align4,.-align4
    .balign 8
align8:
    jump f, t0
    .size align8,.-align8
    .balign 16
align16:
    jump f, t0
    .size align16,.-align16
    .size main, .-main

    .globl f
f:
    ret
    .size f, .-f

# jitlink-check: main = 0x0
# jitlink-check: align2 = 0x2
# jitlink-check: align4 = 0x4
# jitlink-check: align8 = 0x8
# jitlink-check: align16 = 0x10

## main: c.j f
# jitlink-check: (*{2}(main))[1:0] = 0x1
# jitlink-check: (*{2}(main))[15:13] = 0x5
# jitlink-check: decode_operand(main, 0)[11:0] = (f - main)[11:0]

## align2: c.j f
# jitlink-check: (*{2}(align2))[1:0] = 0x1
# jitlink-check: (*{2}(align2))[15:13] = 0x5
# jitlink-check: decode_operand(align2, 0)[11:0] = (f - align2)[11:0]

## align4: c.j f; c.nop
# jitlink-check: (*{2}(align4))[1:0] = 0x1
# jitlink-check: (*{2}(align4))[15:13] = 0x5
# jitlink-check: decode_operand(align4, 0)[11:0] = (f - align4)[11:0]
# jitlink-check: (*{2}(align4+2)) = 0x1

## align8: c.j f; nop; c.nop
# jitlink-check: (*{2}(align8))[1:0] = 0x1
# jitlink-check: (*{2}(align8))[15:13] = 0x5
# jitlink-check: decode_operand(align8, 0)[11:0] = (f - align8)[11:0]
# jitlink-check: (*{4}(align8+2)) = 0x13
# jitlink-check: (*{2}(align8+6)) = 0x1

## align16: c.j f
# jitlink-check: (*{2}(align16))[1:0] = 0x1
# jitlink-check: (*{2}(align16))[15:13] = 0x5
# jitlink-check: decode_operand(align16, 0)[11:0] = (f - align16)[11:0]
