# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/file_to_test.o %S/Inputs/MachO_test_harness_test.s
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/test_harness.o %s
# RUN: not llvm-jitlink -noexec -check %s %t/file_to_test.o \
# RUN:    -harness %t/test_harness.o
# RUN: llvm-jitlink -noexec -phony-externals -check %s %t/file_to_test.o \
# RUN:    -harness %t/test_harness.o
#
# Check that we
#   (1) Can call global symbols in the test object.
#   (2) Can call private symbols in the test object.
#   (3) Can interpose global symbols in the test object.
#   (4) Can interpose private symbols in the test object.
#   (5) Don't need to resolve unused externals in the test object.

.section	__TEXT,__text,regular,pure_instructions

  .globl	_public_func_to_interpose
	.p2align	4, 0x90
_public_func_to_interpose:
	retq

	.globl	_private_func_to_interpose
	.p2align	4, 0x90
_private_func_to_interpose:
	retq

	.globl	_main
	.p2align	4, 0x90
_main:
	callq	_public_func_to_test
	callq	_private_func_to_test
  xorl  %eax, %eax
	retq

	.section	__DATA,__data

# Check that the harness and test file agree on the address of the addresses
# of the interposes:

# jitlink-check: *{8}_public_func_to_interpose_as_seen_by_harness = \
# jitlink-check:   *{8}_public_func_to_interpose_as_seen_by_test

# jitlink-check: *{8}_private_func_to_interpose_as_seen_by_harness = \
# jitlink-check:   *{8}_private_func_to_interpose_as_seen_by_test

  .globl	_public_func_to_interpose_as_seen_by_harness
	.p2align	3
_public_func_to_interpose_as_seen_by_harness:
	.quad	_public_func_to_interpose

	.globl	_private_func_to_interpose_as_seen_by_harness
	.p2align	3
_private_func_to_interpose_as_seen_by_harness:
	.quad	_private_func_to_interpose

# We need to reference the *_as_seen_by_test pointers used above to ensure
# that they're not dead-stripped as unused.
  .globl  _anchor_test_case_pointers
  .p2align  3
_anchor_test_case_pointers:
  .quad _public_func_to_interpose_as_seen_by_test
  .quad _private_func_to_interpose_as_seen_by_test

.subsections_via_symbols
