# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_extra_def_strong.o %S/Inputs/MachO_extra_def_strong.s
# RUN: llvm-ar crs %t/libExtraDef.a %t/MachO_extra_def_strong.o
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_archive_support.o %s
# RUN: llvm-jitlink -noexec %t/MachO_archive_support.o -lFoo -jd Foo -L%t -lExtraDef
# RUN: llvm-jitlink -noexec %t/MachO_archive_support.o -lFoo -jd Foo %t/libExtraDef.a
#
# Test that archives can be loaded and referenced from other JITDylibs.

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
