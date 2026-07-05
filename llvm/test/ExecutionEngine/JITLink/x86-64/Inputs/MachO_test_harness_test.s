	.section	__TEXT,__text,regular,pure_instructions
	.globl	_unused_public_function
	.p2align	4, 0x90
_unused_public_function:
	jmp	_unresolvable_external

	.p2align	4, 0x90
_unused_private_function:
	jmp	_unresolvable_external

	.globl	_public_func_to_interpose
	.p2align	4, 0x90
_public_func_to_interpose:
	retq

  .p2align	4, 0x90
_private_func_to_interpose:
	retq

	.globl	_used_weak
	.weak_definition	_used_weak
	.p2align	4, 0x90
_used_weak:
	retq

	.globl	_public_func_to_test
	.p2align	4, 0x90
_public_func_to_test:
  callq _used_weak
  callq _used_unresolved_external
	jmp	_public_func_to_interpose

	.p2align	4, 0x90
_private_func_to_test:
	jmp	_private_func_to_interpose

	.section	__DATA,__data
	.globl	_public_func_to_interpose_as_seen_by_test
	.p2align	3
_public_func_to_interpose_as_seen_by_test:
	.quad	_public_func_to_interpose

	.globl	_private_func_to_interpose_as_seen_by_test
	.p2align	3
_private_func_to_interpose_as_seen_by_test:
	.quad	_private_func_to_interpose

.subsections_via_symbols
