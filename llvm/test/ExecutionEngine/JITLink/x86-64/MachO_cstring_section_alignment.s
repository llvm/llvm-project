# REQUIRES: asserts
# RUN: llvm-mc -triple=x86_64-apple-macos10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec %t 2>&1 \
# RUN:              | FileCheck %s
#
# Verify that PC-begin candidate symbols have been sorted correctly when adding
# PC-begin edges for FDEs. In this test both _main and _X are at address zero,
# however we expect to select _main over _X as _X is common. If the sorting
# fails we'll trigger an assert in EHFrameEdgeFixer, otherwise this test will
# succeed.
#
# CHECK: Graphifying C-string literal section __TEXT,__cstring
# CHECK:    Created block {{.*}} -- {{.*}}, align = 16, align-ofs = 0 for "abcdefghijklmno"

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0
	.globl	_main
	.p2align	4, 0x90
_main:
	retq

	.section	__TEXT,__cstring,cstring_literals
	.p2align	4
L_.str.1:
	.asciz	"abcdefghijklmno"

.subsections_via_symbols
