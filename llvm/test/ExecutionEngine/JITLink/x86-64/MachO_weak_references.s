# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t/macho_weak_refs.o %s
# RUN: llvm-jitlink -noexec -check-name=jitlink-check-bar-present -abs bar=0x1 -check=%s %t/macho_weak_refs.o
# RUN: llvm-jitlink -noexec -check-name=jitlink-check-bar-absent -check=%s %t/macho_weak_refs.o

# Test weak reference handling by linking with and without a definition of 'bar' available.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14	sdk_version 10, 14
	.globl	_main
	.p2align	4, 0x90
_main:
# jitlink-check-bar-present: *{8}(got_addr(macho_weak_refs.o, bar)) = bar
# jitlink-check-bar-absent: *{8}(got_addr(macho_weak_refs.o, bar)) = 0
	cmpq	$0, bar@GOTPCREL(%rip)

	.weak_reference bar

.subsections_via_symbols
