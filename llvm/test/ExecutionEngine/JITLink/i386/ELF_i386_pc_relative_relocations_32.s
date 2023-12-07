# RUN: llvm-mc -triple=i386-unknown-linux-gnu -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -check %s %t.o
#
# Test ELF 32 bit pc relative relocations

        .text

        .globl  main
        .p2align        4
        .type   main,@function
main:
    retl
        .size   main, .-main


# Tests PC relative relocation for positive offset from PC
# jitlink-check: decode_operand(bar, 0) = foo - next_pc(bar)

        .globl  bar
        .p2align        4
        .type   bar,@function
bar:
    calll foo
    .size       bar, .-bar

        .globl  foo
        .p2align        4
        .type   foo,@function
foo:
    retl
    .size       foo, .-foo


# Tests PC relative relocation for negative offset from PC
# jitlink-check: decode_operand(baz, 0) = fooz - next_pc(baz)
        .globl  fooz
        .p2align        4
        .type   fooz,@function
fooz:
    retl
        .size   fooz, .-fooz

        .globl  baz
        .p2align        4
        .type   baz,@function
baz:
    calll fooz
        .size       baz, .-baz