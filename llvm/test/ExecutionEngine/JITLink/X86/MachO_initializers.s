# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec %t.o

        .section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 14	sdk_version 10, 15

        .globl  _main
_main:
        retq

        .section	__TEXT,__StaticInit,regular,pure_instructions
	.p2align	4, 0x90
_foo:
        retq


	.section	__DATA,__mod_init_func,mod_init_funcs
	.p2align	3
	.quad   _foo

.subsections_via_symbols
