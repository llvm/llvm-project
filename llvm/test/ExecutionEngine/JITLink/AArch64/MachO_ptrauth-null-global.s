# RUN: llvm-mc -triple=arm64e-apple-macosx -filetype=obj -o %t.o %s
# RUN: llvm-jitlink %t.o
#
# REQUIRES: system-darwin && host=arm64{{.*}}
#
# Check that arm64e ptrauth pass preserves nulls.
#
# Testcase derived from:
#   extern void __attribute__((weak_import)) f(void);
#   void (*p) = &f;
#
#   int main(int argc, char *argv[]) {
#     return p ? 1 : 0;
#   }

        .section        __TEXT,__text,regular,pure_instructions
        .globl  _main
        .p2align        2
_main:
        adrp    x8, _p@PAGE
        ldr     x8, [x8, _p@PAGEOFF]
        cmp     x8, #0
        cset    w0, ne
        ret

        .section        __DATA,__data
        .globl  _p
        .p2align        3, 0x0
_p:
        .quad   _f@AUTH(ia,0)

        .weak_reference _f
        .weak_reference l_f.ptrauth
.subsections_via_symbols
