# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=arm64-apple-darwin19 -filetype=obj -o %t/macho_opt.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 64Kb -slab-page-size 16384 -slab-address 0xfff00000 \
# RUN:     -abs _var0=0xfff80008 -abs _var1=0xfe00000c -abs _var2=0xf00000014 \
# RUN:     -abs _func0=0xffff0000 \
# RUN:     -check=%s %t/macho_opt.o

        .section        __TEXT,__text,regular,pure_instructions

# We should branch directly to func0, without the stub.
#
# jitlink-check: decode_operand(_main, 0) = (_func0 - _main)[27:2]
        .globl _main
        .p2align        2
_main:
        bl      _func0
        ret

# Loading var0's address (21 bit range) is optimized to:
#   nop
#   adr x0, _var0
#
# NOP
# jitlink-check: *{4}_load_var0 = 0xd503201f
# ADR x0, ?
# jitlink-check: (*{4}(_load_var0 + 4)) & 0x9f00001f = 0x10000000
# jitlink-check: decode_operand(_load_var0 + 4, 1) = _var0 - (_load_var0+4)
        .globl  _load_var0
        .p2align        2
_load_var0:
        adrp    x1, _var0@GOTPAGE
        ldr     x0, [x1, _var0@GOTPAGEOFF]
        ldr     x0, [x0]
        ret

# Loading var1's address (33 bit range) is optimized to:
#   adrp x1, _var1
#   add  x0, x1, _var1@PAGEOFF
#
# ADRP x1, ?
# jitlink-check: (*{4}_load_var1) & 0x9f00001f = 0x90000001
# jitlink-check: decode_operand(_load_var1, 1) = _var1[32:12] - _load_var1[32:12]
# ADD  x0, x1, ?
# jitlink-check: (*{4}(_load_var1+4)) & 0xffc003ff = 0x91000020
# jitlink-check: decode_operand(_load_var1+4, 2) = _var1[11:0]

        .globl  _load_var1
        .p2align        2
_load_var1:
        adrp    x1, _var1@GOTPAGE
        ldr     x0, [x1, _var1@GOTPAGEOFF]
        ldr     x0, [x0]
        ret

# Loading var2's address (>33 bit range) is not optimized.
#
# ADRP x1, ?
# jitlink-check: (*{4}_load_var2) & 0x9f00001f = 0x90000001
# jitlink-check: decode_operand(_load_var2, 1) = \
# jitlink-check:    (got_addr(macho_opt.o, _var2)[32:12] - _load_var2[32:12])
# LDR  x0, x1, ?
# jitlink-check: (*{4}(_load_var2+4)) & 0xffc003ff = 0xf9400020
# jitlink-check: decode_operand(_load_var2+4, 2) = \
# jitlink-check:    got_addr(macho_opt.o, _var2)[11:3]
        .globl  _load_var2
        .p2align        2
_load_var2:
        adrp    x1, _var2@GOTPAGE
        ldr     x0, [x1, _var2@GOTPAGEOFF]
        ldr     x0, [x0]
        ret

        .subsections_via_symbols
