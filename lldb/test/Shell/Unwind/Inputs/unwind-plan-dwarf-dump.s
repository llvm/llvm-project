        .text
        .globl  main
        .type   main, @function
main:
.LFB0:
        .cfi_startproc
        .cfi_escape 0x0f, 0x05, 0x77, 0x00, 0x08, 0x00, 0x22
        .cfi_escape 0x16, 0x10, 0x04, 0x09, 0xf8, 0x22, 0x06
        movl    $47, %eax
        ret
        .cfi_endproc
.LFE0:
        .size   main, .-main
