# RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-rtdyld -triple=aarch64-none-linux-gnu -execute --entry=foo --preallocate=0 -check=%s %t
# RUN: llvm-rtdyld -triple=aarch64-none-linux-gnu -execute --entry=foo --preallocate=1024 -check=%s %t


       .text
       .globl  foo
       .p2align        2
       .type   foo,@function
foo:
       adrp    x1, .L.str
       add     x1, x1, :lo12:.L.str
        mov     w0, wzr
        ret
.Lfunc_end0:
       .size   foo, .Lfunc_end0-foo
       .type   .L.str,@object
       .section        .rodata.str1.1,"aMS",@progbits,1
.L.str:
       .asciz  "foo"
       .size   .L.str, 4
