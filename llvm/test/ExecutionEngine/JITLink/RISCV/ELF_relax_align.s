## Test that we can handle R_RISCV_ALIGN.

# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o %t.rv32
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x0 -slab-page-size 4096 \
# RUN:     -check %s %t.rv32

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.rv64
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x0 -slab-page-size 4096 \
# RUN:     -check %s %t.rv64

    .globl main,align4,align8,align16,align32
    .size align4, 1
    .size align8, 1
    .size align16, 1
    .size align32, 1
main:
    call f
    .balign 4
align4:
    call f
    .balign 8
align8:
    call f
    .balign 16
align16:
    call f
    .balign 32
align32:
    call f
    .size main, .-main

    .globl f
f:
    ret
    .size f, .-f

# jitlink-check: main = 0x0
# jitlink-check: align4 = 0x4
# jitlink-check: align8 = 0x8
# jitlink-check: align16 = 0x10
# jitlink-check: align32 = 0x20

## main: jal f
# jitlink-check: (*{4}(main))[11:0] = 0xef
# jitlink-check: decode_operand(main, 1) = (f - main)

## align 4: jal f
# jitlink-check: (*{4}(align4))[11:0] = 0xef
# jitlink-check: decode_operand(align4, 1) = (f - align4)

## align8: jal f; nop
# jitlink-check: (*{4}(align8))[11:0] = 0xef
# jitlink-check: decode_operand(align8, 1) = (f - align8)
# jitlink-check: (*{4}(align8+4)) = 0x13

## align16: jal f; nop; nop; nop
# jitlink-check: (*{4}(align16))[11:0] = 0xef
# jitlink-check: decode_operand(align16, 1) = (f - align16)
# jitlink-check: (*{4}(align16+4)) = 0x13
# jitlink-check: (*{4}(align16+8)) = 0x13
# jitlink-check: (*{4}(align16+12)) = 0x13

## align32: jal f
# jitlink-check: (*{4}(align32))[11:0] = 0xef
# jitlink-check: decode_operand(align32, 1) = (f - align32)
