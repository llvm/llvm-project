# Supplies a global definition, l_foo, with a linker-private prefix. Since this
# definition is marked as global it should be resolvable outside the object.

	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 14
        .globl          l_foo
	.p2align	4, 0x90
l_foo:
	xorl	%eax, %eax
	retq

.subsections_via_symbols
