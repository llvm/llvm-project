# RUN: llvm-mc -triple=x86_64-apple-macos10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec -check=%s %t
#
# Check that __thread_bss sections are handled as zero-fill.
#
# jitlink-check: *{4}X = 0

        .section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 15	sdk_version 10, 15
	.globl	_main
	.p2align	4, 0x90
_main:
        retq

        .globl X
.tbss X, 4, 2


.subsections_via_symbols
