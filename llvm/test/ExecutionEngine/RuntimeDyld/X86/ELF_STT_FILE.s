# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -o %T/test_ELF_STT_FILE_FILE_x86-64.o %p/Inputs/ELF_STT_FILE_FILE.s
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -o %T/test_ELF_STT_FILE_GLOBAL_x86-64.o %p/Inputs/ELF_STT_FILE_GLOBAL.s
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -o %T/test_ELF_STT_FILE_x86-64.o %s
# RUN: llvm-rtdyld -triple=x86_64-pc-linux -verify %T/test_ELF_STT_FILE_GLOBAL_x86-64.o %T/test_ELF_STT_FILE_FILE_x86-64.o %T/test_ELF_STT_FILE_x86-64.o

# Test that RTDyldELF ignores STT_FILE symbols, and in particular does
# crash if we are relocating against a symbol that happens to have the
# same name as an STT_FILE symbol.

_main:
    movq    foo.c@GOTPCREL(%rip), %rax
    movq    bar.c@GOTPCREL(%rip), %rax
    movq    $0, %rax
    retq
