# RUN: llvm-mc -triple=aarch64_be-none-linux-gnu -filetype=obj -o %T/be-reloc.o %s
# RUN: llvm-rtdyld -triple=aarch64_be-none-linux-gnu -verify -dummy-extern f=0x0123456789abcdef -check=%s %T/be-reloc.o

        .globl Q
        .section .dummy, "ax"
Q:
        nop

        .text
        .globl  g
        .p2align        2
        .type   g,@function
g:
# R_AARCH64_MOVW_UABS_G3
        movz    x0, #:abs_g3:f
# R_AARCH64_MOVW_UABS_G2_NC
        movk    x0, #:abs_g2_nc:f
# R_AARCH64_MOVW_UABS_G1_NC
        movk    x0, #:abs_g1_nc:f
# R_AARCH64_MOVW_UABS_G0_NC
        movk    x0, #:abs_g0_nc:f
        ret
        .Lfunc_end0:
        .size   g, .Lfunc_end0-g

        .type   k,@object
        .data
        .globl  k
        .p2align        3
k:
        .xword  f        
        .size   k, 8
r:
# R_AARCH64_PREL32: use Q instead of f to fit in 32 bits.
        .word  Q - .
# R_AARCH64_PREL64
        .p2align        3
        .xword f - .

# LE instructions read as BE
# rtdyld-check: *{4}(g) = 0x6024e0d2
# rtdyld-check: *{4}(g + 4) = 0xe0acc8f2
# rtdyld-check: *{4}(g + 8) = 0x6035b1f2
# rtdyld-check: *{4}(g + 12) = 0xe0bd99f2
# rtdyld-check: *{8}k = f
# rtdyld-check: *{4}r = (Q - r)[31:0]
# rtdyld-check: *{8}(r + 8) = f - r - 8
