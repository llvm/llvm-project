# This file deliberately has a line table and symbols, but no debug-info DIEs.
        .file 1 "/tmp/standalone.c"

        .text
        .globl foo
        .type foo,@function
foo:
        .loc 1 42 7
        nop
        .loc 1 43 3
        nop
        retq
.Lfoo_end:
        .size foo, .Lfoo_end-foo
