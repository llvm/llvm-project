# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_extra_def_strong.o %S/Inputs/MachO_extra_def_strong.s
# RUN: llvm-ar crs %t/libExtraDef.a %t/MachO_extra_def_strong.o
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_archive_load_hidden_support.o %s
# RUN: not llvm-jitlink -noexec %t/MachO_archive_load_hidden_support.o -lFoo \
# RUN:   -jd Foo -L%t -hidden-lExtraDef
#
# Expect this test to fail when we try to reference ExtraDef, which should have
# be hidden in JITDylib Foo. This tests that we're correctly overriding the
# object interface when we load the object.

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
