# RUN: llvm-mc -triple=arm64-none-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-rtdyld -triple=arm64-none-linux-gnu -verify -dummy-extern f=0x0123456789abcdef -check=%s %t
        
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
l:
# R_AARCH64_LDST8_ABS_LO12_NC
        ldrsb x4, [x5, :lo12:a+1]
# R_AARCH64_LDST16_ABS_LO12_NC
        ldrh w4, [x5, :lo12:a+2]
# R_AARCH64_LDST32_ABS_LO12_NC
        ldr s4, [x5, :lo12:a]
# R_AARCH64_LDST64_ABS_LO12_NC
        ldr x4, [x5, :lo12:a]
# R_AARCH64_LDST128_ABS_LO12_NC
        ldr q4, [x5, :lo12:a]
p:
# R_AARCH64_ADR_PREL_PG_HI21
# Test both low and high immediate values
        adrp x4, a + 20480 // 16384 + 4096
# Align next label to 16 bytes, so that LDST immediate
# fields will be non-zero        
        .align 4
a:
# R_AARCH64_ADD_ABS_LO12_NC
        add x0, x0, :lo12:f
        ret
        .Lfunc_end0:
        .size   g, .Lfunc_end0-g

        .type   k,@object
        .data
        .globl  k
        .p2align        3
k:
        .xword  f
        .size   k, 16
r:
# R_AARCH64_PREL32: use Q instead of f to fit in 32 bits.
        .word  Q - .        
# R_AARCH64_PREL64
        .p2align        3
        .xword f - .

# rtdyld-check: *{4}(g) = 0xd2e02460
# rtdyld-check: *{4}(g + 4) = 0xf2c8ace0
# rtdyld-check: *{4}(g + 8) = 0xf2b13560
# rtdyld-check: *{4}(g + 12) = 0xf299bde0

## Check LDSTXX_ABS_LO12_NC
# rtdyld-check: (*{4}l)[21:10] = (a+1)[11:0]
# rtdyld-check: (*{4}(l+4))[21:10] = (a+2)[11:1]
# rtdyld-check: (*{4}(l+8))[21:10] = a[11:2]
# rtdyld-check: (*{4}(l+12))[21:10] = a[11:3]
# rtdyld-check: (*{4}(l+16))[21:10] = a[11:4]

## Check ADR_PREL_PG_HI21. Low order bits of immediate value
## go to bits 30:29. High order bits go to bits 23:5
# rtdyld-check: (*{4}p)[30:29] = (a - p + 20480)[13:12]
# rtdyld-check: (*{4}p)[23:5] = (a - p + 20480)[32:14]

# rtdyld-check: *{8}k = f
# rtdyld-check: *{4}r = (Q - r)[31:0]
# rtdyld-check: *{8}(r + 8) = f - r - 8

## f & 0xFFF = 0xdef (bits 11:0 of f)
## 0xdef << 10 = 0x37bc00
# rtdyld-check: *{4}(a) = 0x9137bc00
