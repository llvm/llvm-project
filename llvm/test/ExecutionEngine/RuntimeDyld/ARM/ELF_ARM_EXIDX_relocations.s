# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=arm-linux-gnueabihf -filetype=obj -o %t/reloc.o %s
# RUN: llvm-rtdyld -triple=arm-linux-gnueabihf -verify -map-section reloc.o,.ARM.exidx=0x6000 -map-section reloc.o,.text=0x4000  -dummy-extern __aeabi_unwind_cpp_pr0=0x1234 -check=%s %t/reloc.o

        .text
        .syntax unified
        .eabi_attribute 67, "2.09"      @ Tag_conformance
        .cpu    cortex-a8
        .fpu    neon
        .file   "reloc.c"
        .globl  g
        .align  2
        .type   g,%function
g:
        .fnstart
        movw    r0, #1
        bx      lr
        .Lfunc_end0:
        .size   g, .Lfunc_end0-g
        .fnend

# rtdyld-check: *{4}(section_addr(reloc.o, .ARM.exidx)) = (g - (section_addr(reloc.o, .ARM.exidx))) & 0x7fffffff
# Compat unwind info: finish(0xb0), finish(0xb0), finish(0xb0)
# rtdyld-check: *{4}(section_addr(reloc.o, .ARM.exidx) + 0x4) = 0x80b0b0b0
