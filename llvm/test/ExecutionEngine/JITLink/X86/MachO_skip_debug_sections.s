# REQUIRES: asserts
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -debug-only=jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check that debug sections are not emitted, and consequently that we don't
# error out due to buggy past-the-end anonymous relocations in __debug_ranges.
#
# CHECK: __debug_ranges is a debug section: No graph section will be created.
  .section	__TEXT,__text,regular,pure_instructions
  .macosx_version_min 10, 15
	.globl	_main
	.p2align	4, 0x90
_main:
	retq
Lpast_the_end:

	.section	__DWARF,__debug_ranges
	.p2align	4
	.quad	Lpast_the_end

.subsections_via_symbols
