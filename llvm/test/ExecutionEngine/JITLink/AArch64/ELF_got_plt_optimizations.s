# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=aarch64-linux-gnu -filetype=obj -o %t/elf_opt.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 64Kb -slab-page-size 16384 -slab-address 0xfff00000 \
# RUN:     -abs var0=0xfff80008 -abs var1=0xfe00000c -abs var2=0xf00000014 \
# RUN:     -abs func0=0xffff0000 \
# RUN:     -check=%s %t/elf_opt.o

# We should branch directly to func0, without the PLT.
#
# jitlink-check: decode_operand(main, 0) = (func0 - main)[27:2]
        .globl main
        .p2align        2
main:
        bl      func0
        ret
        .size main, .-main

# Loading var0's address (21 bit range) is optimized to:
#   nop
#   adr x0, var0
#
# NOP
# jitlink-check: *{4}load_var0 = 0xd503201f
# ADR x1, ?
# jitlink-check: (*{4}(load_var0 + 4)) & 0x9f00001f = 0x10000001
# jitlink-check: decode_operand(load_var0 + 4, 1) = var0 - (load_var0+4)
        .globl  load_var0
        .p2align        2
load_var0:
        adrp    x1, :got: var0
        ldr     x1, [x1, :got_lo12: var0]
        ldr     x0, [x1]
        ret
        .size load_var0, .-load_var0

# Loading var1's address (33 bit range) is optimized to:
#   adrp x1, var1
#   add  x1, x1, :lo12:var0
#
# ADRP x1, ?
# jitlink-check: (*{4}load_var1) & 0x9f00001f = 0x90000001
# jitlink-check: decode_operand(load_var1, 1) = var1[64:12]-load_var1[64:12]
# ADD  x1, x1, ?
# jitlink-check: (*{4}(load_var1+4)) & 0xffc003ff = 0x91000021
# jitlink-check: decode_operand(load_var1+4, 2) = var1[11:0]

        .globl  load_var1
        .p2align        2
load_var1:
        adrp    x1, :got: var1
        ldr     x1, [x1, :got_lo12: var1]
        ldr     x0, [x1]
        ret
        .size load_var1, .-load_var1

# Loading var2's address (>33 bit range) is not optimized.
#
# ADRP x1, ?
# jitlink-check: (*{4}load_var2) & 0x9f00001f = 0x90000001
# jitlink-check: decode_operand(load_var2, 1) = \
# jitlink-check:    got_addr(elf_opt.o, var2)[64:12]-load_var2[64:12]
# LDR  x1, x1, ?
# jitlink-check: (*{4}(load_var2+4)) & 0xffc003ff = 0xf9400021
# jitlink-check: decode_operand(load_var2+4, 2) = \
# jitlink-check:    got_addr(elf_opt.o, var2)[11:0] >> 3
        .globl  load_var2
        .p2align        2
load_var2:
        adrp    x1, :got: var2
        ldr     x1, [x1, :got_lo12: var2]
        ldr     x0, [x1]
        ret
        .size load_var2, .-load_var2
