# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_extra_def_strong.o %S/Inputs/MachO_extra_def_strong.s
# RUN: llvm-ar crs %t/libExtraDef.a %t/MachO_extra_def_strong.o
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_archive_load_hidden_support.o %s
# RUN: llvm-jitlink -noexec %t/MachO_archive_load_hidden_support.o \
# RUN:   -L%t -hidden-lExtraDef
# RUN: llvm-jitlink -noexec %t/MachO_archive_load_hidden_support.o \
# RUN:   -load_hidden %t/libExtraDef.a
#
# Expect this test to succeed -- ExtraDef should be hidden, but visible to
# ExtraDefRef as they're linked in the same JITDylib. This tests that we're
# correctly handling the change in object interface when linking ExtraDef.

        .section  __TEXT,__text,regular,pure_instructions

        .globl  _main
        .p2align  4, 0x90
_main:
        retq

	.section	__DATA,__data
	.globl	ExtraDefRef
	.p2align	3
ExtraDefRef:
	.quad	ExtraDef

        .subsections_via_symbols
