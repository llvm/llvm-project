# RUN: llvm-mc -triple=arm64-none-linux-gnu -filetype=obj -o %T/reloc.o %s
# RUN: llvm-rtdyld -triple=arm64-none-linux-gnu -verify -dummy-extern f=0x0123456789abcdef -check=%s %T/reloc.o

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

# rtdyld-check: *{4}(g) = 0xd2e02460
# rtdyld-check: *{4}(g + 4) = 0xf2c8ace0
# rtdyld-check: *{4}(g + 8) = 0xf2b13560
# rtdyld-check: *{4}(g + 12) = 0xf299bde0
# rtdyld-check: *{8}k = f
