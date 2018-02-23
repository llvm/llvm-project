# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -o %T/file.o %p/Inputs/ELF_STT_FILE_GLOBAL.s
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -o %T/relaxed.o %s
# RUN: llvm-rtdyld -triple=x86_64-pc-linux -verify %T/file.o %T/relaxed.o

# Test that RTDyldELF does not crash with 'unimplemented relocation'

_main:
    movq    foo.c@GOTPCREL(%rip), %rax
