# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc --filetype=obj --triple=loongarch32 -mattr=+relax %s -o %t.la32
# RUN: llvm-jitlink --noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x0 -slab-page-size 16384 \
# RUN:     --check %s %t.la32

# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax %s -o %t.la64
# RUN: llvm-jitlink --noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x0 -slab-page-size 16384 \
# RUN:     --check %s %t.la64

## Test that we can handle R_LARCH_ALIGN.

    .text

    .globl main,align4,align8,align16,align32,alignmax12,alignmax8
    .type  main,@function
main:
    bl f
    .align 2
align4:
    bl f
    .size align4, .-align4
    .align 3
align8:
    bl f
    .size align8, .-align8
    .align 4
align16:
    bl f
    .size align16, .-align16
    .align 5
align32:
    bl f
    .size align32, .-align32
    .align 4,,12
alignmax12:
    bl f
    .size alignmax12, .-alignmax12
    .align 4,,8
alignmax8:
    bl f
    .size alignmax8, .-alignmax8
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
# jitlink-check: alignmax12 = 0x30
## 3 nops (12 bytes) should be inserted to satisfy alignment.
## But the max bytes we can insert is 8. So alignment is ignored.
# jitlink-check: alignmax8 = 0x34

## main: bl f
# jitlink-check: (*{4}(main))[31:26] = 0x15
# jitlink-check: decode_operand(main, 0)[27:0] = (f - main)[27:0]

## align 4: bl f
# jitlink-check: (*{4}(align4))[31:26] = 0x15
# jitlink-check: decode_operand(align4, 0)[27:0] = (f - align4)[27:0]

## align8: bl f; nop
# jitlink-check: (*{4}(align8))[31:26] = 0x15
# jitlink-check: decode_operand(align8, 0)[27:0] = (f - align8)[27:0]
# jitlink-check: (*{4}(align8+4)) = 0x3400000

## align16: bl f; nop; nop; nop
# jitlink-check: (*{4}(align16))[31:26] = 0x15
# jitlink-check: decode_operand(align16, 0)[27:0] = (f - align16)[27:0]
# jitlink-check: (*{4}(align16+4)) = 0x3400000
# jitlink-check: (*{4}(align16+8)) = 0x3400000
# jitlink-check: (*{4}(align16+12)) = 0x3400000

## align32: bl f; nop; nop; nop
# jitlink-check: (*{4}(align32))[31:26] = 0x15
# jitlink-check: decode_operand(align32, 0)[27:0] = (f - align32)[27:0]
# jitlink-check: (*{4}(align32+4)) = 0x3400000
# jitlink-check: (*{4}(align32+8)) = 0x3400000
# jitlink-check: (*{4}(align32+12)) = 0x3400000

## alignmax12: bl f
# jitlink-check: (*{4}(alignmax12))[31:26] = 0x15
# jitlink-check: decode_operand(alignmax12, 0)[27:0] = (f - alignmax12)[27:0]

## alignmax8: bl f
# jitlink-check: (*{4}(alignmax8))[31:26] = 0x15
# jitlink-check: decode_operand(alignmax8, 0)[27:0] = (f - alignmax8)[27:0]
