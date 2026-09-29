        .section        __TEXT,__text,regular,pure_instructions
        .globl  _main
        .p2align        2
_main:
        .cfi_startproc
Lloh0:
        adrp    x0, _foo@GOTPAGE
Lloh1:
        ldr     x0, [x0, _foo@GOTPAGEOFF]

        ret
        .loh AdrpLdrGot Lloh0, Lloh1
        .cfi_endproc

.subsections_via_symbols
