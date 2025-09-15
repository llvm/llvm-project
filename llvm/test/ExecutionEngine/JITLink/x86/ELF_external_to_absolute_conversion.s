# RUN: llvm-mc -triple=i386-unknown-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec %t
#
# Check that symbol scope is demoted to local when external symbols are
# converted to absolutes. This is demotion is necessary to avoid "unexpected
# definition" errors.
#
# The reference to _GLOBAL_OFFSET_TABLE_ will trigger creation of an external
# _GLOBAL_OFFSET_TABLE_ symbol, and the GOTOFF relocation will force creation
# of a GOT symbol without actually introducing any GOT entries. Together these
# should cause the external _GLOBAL_OFFSET_TABLE_ symbol to be converted to an
# absolute symbol with address zero. If the scope is not demoted correctly this
# will trigger an "unexpected definition" error.

        .text
        .globl  main                            
        .p2align        4, 0x90
        .type   main,@function
main:                    
        pushl   %ebp
        movl    %esp, %ebp
        pushl   %eax
        calll   .L0$pb
.L0$pb:
        popl    %eax
.Ltmp0:
        addl    $_GLOBAL_OFFSET_TABLE_+(.Ltmp0-.L0$pb), %eax
        movl    $0, -4(%ebp)
        movl    a@GOTOFF(%eax), %eax
        addl    $4, %esp
        popl    %ebp
        retl
        .size   main, .-main


        .type   a,@object                       # @a
        .data
        .p2align        2
a:
        .long   42                              # 0x2a
        .size   a, 4