# Check that the generated __inits symbol name does not clash between objects
# with the same base name in two different static archives. Otherwise we get a
# duplicate symbol error.

# RUN: rm -rf %t && mkdir -p %t
# RUN: split-file %s %t

# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/dir1/myobj.o %t/dir1/myobj.s
# RUN: llvm-ar crs %t/libmyobj1.a %t/dir1/myobj.o

# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/dir2/myobj.o %t/dir2/myobj.s
# RUN: llvm-ar crs %t/libmyobj2.a %t/dir2/myobj.o

# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/main.o %t/main.s

# RUN: llvm-jitlink -noexec %t/main.o -lmyobj1 -lmyobj2 -L%t

#--- dir1/myobj.s
        .section        __TEXT,__text,regular,pure_instructions
        .build_version macos, 15, 0     sdk_version 15, 0
        .globl  _myobj1
        .p2align        4, 0x90
_myobj1:                                     ## @f
        retq

        .section        __DATA,__mod_init_func,mod_init_funcs
        .p2align        3, 0x0
        .quad   _myobj1

        .subsections_via_symbols

#--- dir2/myobj.s
        .section        __TEXT,__text,regular,pure_instructions
        .build_version macos, 15, 0     sdk_version 15, 0
        .globl  _myobj2
        .p2align        4, 0x90
_myobj2:                                     ## @f
        retq

        .section        __DATA,__mod_init_func,mod_init_funcs
        .p2align        3, 0x0
        .quad   _myobj2

        .subsections_via_symbols

#--- main.s

        .section  __TEXT,__text,regular,pure_instructions

        .globl  _main
        .p2align  4, 0x90
_main:
        pushq   %rbp
        movq    %rsp, %rbp
        callq   _myobj1
        callq   _myobj2
        xorl    %eax, %eax
        popq    %rbp
        retq

        .subsections_via_symbols
